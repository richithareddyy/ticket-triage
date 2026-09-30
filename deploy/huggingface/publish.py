"""Publish the demo to a Hugging Face Space (free Docker hosting).

    hf auth login                                   # once; paste a token with "write" access
    python deploy/huggingface/publish.py --space <your-username>/ticket-triage

Creates the Space if it doesn't exist, then uploads only what the demo needs:
the Dockerfile (its last stage, `space`, runs the API + UI), the code and the
trained model. Hugging Face builds the image and serves it on a public URL.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder, whoami

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ["Dockerfile", "requirements.txt", "requirements-ui.txt", "triage", "api", "ui",
           "deploy/start-space.sh", "artifacts/model.joblib", "artifacts/metrics.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="<username>/<space-name>")
    ap.add_argument("--stage-only", metavar="DIR", help="assemble the upload folder into DIR and stop (for testing)")
    ap.add_argument("--github-url", help="optional link to the source repo, added to the Space card")
    args = ap.parse_args()

    missing = [p for p in INCLUDE if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)} (run `make train` first?)")
    if not (args.space or args.stage_only):
        ap.error("--space is required")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(args.stage_only or tmp)
        for rel in INCLUDE:
            src, dst = ROOT / rel, stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)
        card = (ROOT / "deploy/huggingface/SPACE_README.md").read_text()
        if args.github_url:
            card += f"\nSource code: {args.github_url}\n"
        (stage / "README.md").write_text(card)
        if args.stage_only:
            print(f"staged upload folder at {stage}")
            return

        print(f"signed in to Hugging Face as {whoami()['name']}")
        url = create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        print(f"uploading to {url}")
        upload_folder(repo_id=args.space, repo_type="space", folder_path=stage,
                      commit_message="Deploy ticket triage demo")

    owner, name = args.space.split("/", 1)
    print(f"\nDone. Build logs: https://huggingface.co/spaces/{args.space}")
    print(f"Live app (after the build finishes, ~5 min): https://{owner}-{name}.hf.space".lower())


if __name__ == "__main__":
    main()
