import os, json
import kagglehub

out = {"errors": []}
for handle in ["devdope/900k-spotify", "carlosgdcj/genius-song-lyrics-with-language-information"]:
    try:
        p = kagglehub.dataset_download(handle)
        files = []
        for root, _, fs in os.walk(p):
            for f in fs:
                rel = os.path.relpath(os.path.join(root, f), p)
                files.append(rel)
        out[handle] = {
            "download_path": p,
            "file_count": len(files),
            "files": files[:40]
        }
    except Exception as e:
        out["errors"].append(f"{handle}: {e}")

print(json.dumps(out, indent=2))
