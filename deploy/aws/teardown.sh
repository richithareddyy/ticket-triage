#!/usr/bin/env bash
# Delete the CloudFormation stack and the ECR repositories (stops all AWS charges
# from this project).
#
#   AWS_REGION=us-east-1 ./deploy/aws/teardown.sh
set -euo pipefail

REGION="${AWS_REGION:?set AWS_REGION, e.g. us-east-1}"
STACK_NAME="${STACK_NAME:-ticket-triage}"

read -r -p "Delete stack '$STACK_NAME' and ECR repos ${STACK_NAME}-api/${STACK_NAME}-ui in $REGION? [y/N] " ok
[ "$ok" = "y" ] || { echo "aborted"; exit 1; }

aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK_NAME"
echo "==> Waiting for stack deletion"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK_NAME"

for target in api ui; do
  aws ecr delete-repository --region "$REGION" --repository-name "${STACK_NAME}-${target}" --force >/dev/null 2>&1 &&
    echo "deleted ECR repo ${STACK_NAME}-${target}" || true
done
