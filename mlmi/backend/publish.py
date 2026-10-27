"""Publish a built Sheep file (.ssf) to the signed-in person's Hugging Face account, so others can
add it ready to run, with no build (Part C5, 2026-10-21).

Only models whose licence allows sharing copies are published, with the notices their licence
asks for. A new repository starts private; its owner makes it public on Hugging Face.
"""
import json
import re
from pathlib import Path

# licences that allow redistribution of the weights (with the notice below, where one is asked for)
OPEN = re.compile(r"^(apache-2\.0|mit|bsd-[23]-clause|cc-by-4\.0|cc-by-sa-4\.0|openrail.*|llama3(\.[123])?|"
                  r"gemma|nvidia-open-model-license)$", re.I)
NOTICE = {"llama3": "Built with Llama. Llama is licensed under its community license, Copyright © Meta Platforms, Inc.",
          "gemma": "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms."}


def licence_of(entry, source_dir):
    lic = str((entry or {}).get("licence") or "").strip().lower()
    if not lic and source_dir:
        readme = Path(source_dir) / "README.md"
        try:
            m = re.search(r"^license:\s*(\S+)", readme.read_text(encoding="utf-8", errors="replace"), re.M)
            lic = m.group(1).strip().lower() if m else ""
        except OSError:
            lic = ""
    return lic


def can_share(lic):
    """(ok, why)."""
    if not lic:
        return False, "the model's licence is unknown, so Sheep will not share it"
    if not OPEN.match(lic):
        return False, f"the model's licence ({lic}) does not clearly allow sharing copies"
    return True, ""


def card(path, entry, lic, source):
    notice = next((v for k, v in NOTICE.items() if lic.startswith(k)), "")
    return (f"---\nlicense: {lic}\ntags:\n- sheep\n- ssf\n---\n\n# {Path(path).stem} (Sheep file)\n\n"
            f"A ready-to-run Sheep model built from [{source}](https://huggingface.co/{source}) "
            f"at the {entry.get('built', {}).get('level', 'balanced')} level. Add it in Sheep with this page's ID: "
            "it downloads and runs with no build.\n\n"
            f"The original model's licence applies ({lic}).{(' ' + notice) if notice else ''}\n")


def publish(path, entry, name="", private=True):
    """Upload the .ssf and a model card. Returns {"repo", "url"}. Raises ValueError with the reason."""
    from huggingface_hub import HfApi, get_token
    from .hf_source import signed_in
    p = Path(path)
    if p.suffix.lower() != ".ssf" or not p.is_file():
        raise ValueError("only a built Sheep file (.ssf) can be published")
    if not get_token():
        raise ValueError("sign in to Hugging Face first (a token with write access)")
    user = signed_in()
    source = str((entry or {}).get("hf_repo") or "")
    lic = licence_of(entry, (entry or {}).get("source_dir"))
    ok, why = can_share(lic)
    if not ok:
        raise ValueError(why)
    repo = f"{user}/{re.sub(r'[^A-Za-z0-9._-]+', '-', name or (p.stem + '-sheep'))}"
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=card(p, entry or {}, lic, source or p.stem).encode("utf-8"),
                    path_in_repo="README.md", repo_id=repo)
    lic_file = Path(str((entry or {}).get("source_dir") or "")) / "LICENSE"
    if lic_file.is_file():
        api.upload_file(path_or_fileobj=str(lic_file), path_in_repo="LICENSE", repo_id=repo)
    api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=repo)
    return {"repo": repo, "url": f"https://huggingface.co/{repo}", "private": private}


def ssf_on_page(names):
    """The .ssf file on a Hugging Face page, or None: such a page is added ready to run."""
    files = [n for n in names if n.lower().endswith(".ssf") and "/" not in n]
    return sorted(files, key=len)[0] if files else None
