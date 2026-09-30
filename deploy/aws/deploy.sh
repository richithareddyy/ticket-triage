#!/usr/bin/env bash
# Build the API and UI images for linux/amd64, push them to ECR, and deploy the
# CloudFormation stack (ECS Fargate + Application Load Balancer).
#
#   AWS_REGION=us-east-1 ./deploy/aws/deploy.sh
#
# Optional env vars:
#   STACK_NAME    (default: ticket-triage)
#   IMAGE_TAG     (default: current git SHA, or a timestamp)
#   VPC_ID        (default: the account's default VPC)
#   SUBNET_IDS    (comma-separated; default: the default VPC's subnets)
#   ALLOWED_CIDR  (default: 0.0.0.0/0 - set to "$(curl -s ifconfig.me)/32" to restrict)
#
# Requires: aws CLI v2 (configured credentials), docker with buildx.
set -euo pipefail

cd "$(dirname "$0")/../.."

REGION="${AWS_REGION:?set AWS_REGION, e.g. us-east-1}"
STACK_NAME="${STACK_NAME:-ticket-triage}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
ALLOWED_CIDR="${ALLOWED_CIDR:-0.0.0.0/0}"

[ -f artifacts/model.joblib ] || { echo "artifacts/model.joblib missing - run 'make train' first" >&2; exit 1; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

VPC_ID="${VPC_ID:-$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)}"
[ "$VPC_ID" != "None" ] || { echo "No default VPC found; set VPC_ID and SUBNET_IDS" >&2; exit 1; }
SUBNET_IDS="${SUBNET_IDS:-$(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')}"

echo "==> Logging in to ECR ($REGISTRY)"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

for target in api ui; do
  repo="${STACK_NAME}-${target}"
  aws ecr describe-repositories --region "$REGION" --repository-names "$repo" >/dev/null 2>&1 ||
    aws ecr create-repository --region "$REGION" --repository-name "$repo" \
      --image-scanning-configuration scanOnPush=true >/dev/null
  echo "==> Building and pushing $repo:$IMAGE_TAG (linux/amd64)"
  # Fargate runs x86_64 here; building explicitly avoids pushing arm64 images from Apple Silicon.
  docker buildx build --platform linux/amd64 --target "$target" \
    -t "$REGISTRY/$repo:$IMAGE_TAG" --push .
done

echo "==> Deploying CloudFormation stack $STACK_NAME"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK_NAME" \
  --template-file deploy/aws/cloudformation.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ProjectName="$STACK_NAME" VpcId="$VPC_ID" SubnetIds="$SUBNET_IDS" AllowedCidr="$ALLOWED_CIDR" \
    ApiImage="$REGISTRY/${STACK_NAME}-api:$IMAGE_TAG" UiImage="$REGISTRY/${STACK_NAME}-ui:$IMAGE_TAG"

aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
