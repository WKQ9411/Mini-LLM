import argparse
import importlib
import importlib.util
import json
import platform
import sys
import urllib.request


def print_json(value: dict) -> None:
    print(json.dumps(value))


def get_torch_abi(torch) -> bool:
    abi_fn = getattr(torch, "compiled_with_cxx11_abi", None)
    return bool(abi_fn()) if abi_fn else bool(torch._C._GLIBCXX_USE_CXX11_ABI)


def module_available(module_name: str) -> int:
    try:
        importlib.import_module(module_name)
        return 0
    except Exception:
        return 1


def collect_status() -> dict:
    result = {"python": ".".join(map(str, sys.version_info[:3]))}
    for module_name in ("torch", "triton", "flash_attn"):
        if importlib.util.find_spec(module_name) is None:
            result[module_name] = {
                "installed": False,
                "version": None,
                "available": False,
            }
            continue
        try:
            module = importlib.import_module(module_name)
            result[module_name] = {
                "installed": True,
                "version": getattr(module, "__version__", "unknown"),
                "available": True,
            }
        except Exception:
            result[module_name] = {
                "installed": True,
                "version": None,
                "available": False,
            }
    return result


def show_status(output_format: str) -> int:
    result = collect_status()
    if output_format == "shell":
        print(f"python\t{result['python']}")
        for module_name in ("torch", "triton", "flash_attn"):
            module = result[module_name]
            print(
                "\t".join(
                    (
                        module_name,
                        "1" if module["installed"] else "0",
                        "1" if module["available"] else "0",
                        str(module["version"] or ""),
                    )
                )
            )
    else:
        print_json(result)
    return 0


def verify_main() -> int:
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            import triton

            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Triton: {triton.__version__}")
        return 0
    except Exception as exc:
        print(f"Main environment verification failed: {exc}", file=sys.stderr)
        return 1


def collect_flash_environment() -> dict:
    result = {"ready": False}
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        if torch.version.cuda is None:
            raise RuntimeError("the installed PyTorch build has no CUDA runtime")
        capability = torch.cuda.get_device_capability(0)
        if capability < (8, 0):
            raise RuntimeError(
                f"GPU compute capability {capability[0]}.{capability[1]} is below 8.0"
            )
        if int(torch.version.cuda.split(".")[0]) < 12:
            raise RuntimeError(
                f"CUDA {torch.version.cuda} is below the supported CUDA 12 runtime"
            )
        result.update(
            {
                "ready": True,
                "python": platform.python_version(),
                "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
                "torch": torch.__version__,
                "torch_mm": ".".join(torch.__version__.split("+")[0].split(".")[:2]),
                "cuda": torch.version.cuda,
                "cuda_major": torch.version.cuda.split(".")[0],
                "abi": get_torch_abi(torch),
                "gpu": torch.cuda.get_device_name(0),
                "capability": f"{capability[0]}.{capability[1]}",
                "platform": sys.platform,
                "machine": platform.machine(),
            }
        )
    except Exception as exc:
        result["reason"] = str(exc)
    return result


def show_flash_environment(output_format: str) -> int:
    result = collect_flash_environment()
    if output_format == "shell":
        if result["ready"]:
            fields = (
                "READY",
                result["python"],
                result["python_tag"],
                result["torch"],
                result["torch_mm"],
                result["cuda"],
                result["cuda_major"],
                str(result["abi"]),
                result["gpu"],
                result["capability"],
                result["platform"],
                result["machine"],
            )
        else:
            fields = ("UNSUPPORTED", result.get("reason", "unknown reason"))
        print("\t".join(fields))
    else:
        print_json(result)
    return 0


def collect_flash_wheel() -> dict:
    try:
        import torch

        if torch.version.cuda is None:
            raise RuntimeError("the installed PyTorch build has no CUDA runtime")
        torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        cuda_major = torch.version.cuda.split(".")[0]
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        machine = platform.machine().lower()
        if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
            platform_tag = "win_amd64"
        elif sys.platform.startswith("linux") and machine in {"amd64", "x86_64"}:
            platform_tag = "linux_x86_64"
        else:
            raise RuntimeError(
                f"no prebuilt-wheel platform mapping for {sys.platform}/{machine}"
            )

        request = urllib.request.Request(
            "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=50",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Mini-LLM-setup",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            releases = json.load(response)

        build_tag = (
            f"+cu{cuda_major}torch{torch_mm}cxx11abi"
            f"{str(get_torch_abi(torch)).upper()}"
        ).lower()
        wheel_suffix = f"-{python_tag}-{python_tag}-{platform_tag}.whl".lower()
        for release in releases:
            if release.get("draft") or release.get("prerelease"):
                continue
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                lowered = name.lower()
                if (
                    lowered.startswith("flash_attn-2")
                    and build_tag in lowered
                    and lowered.endswith(wheel_suffix)
                ):
                    return {
                        "found": True,
                        "name": name,
                        "url": asset["browser_download_url"],
                        "release": release.get("tag_name", ""),
                    }
        return {
            "found": False,
            "reason": f"no release asset matched {build_tag}*{wheel_suffix}",
        }
    except Exception as exc:
        return {"found": False, "error": str(exc)}


def find_flash_wheel(output_format: str) -> int:
    result = collect_flash_wheel()
    if output_format == "shell":
        if result["found"]:
            fields = ("FOUND", result["name"], result["url"], result["release"])
        elif "error" in result:
            fields = ("ERROR", result["error"])
        else:
            fields = ("NOT_FOUND", result["reason"])
        print("\t".join(fields))
    else:
        print_json(result)
    return 0


def test_flash_attention() -> int:
    try:
        import torch
        import flash_attn
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        q = torch.randn(8, 4, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
        out = flash_attn_varlen_func(q, k, v, cu, cu, 8, 8, causal=True)
        torch.cuda.synchronize()
        if out.shape != q.shape or not torch.isfinite(out).all():
            raise RuntimeError("flash-attn returned an invalid output")
        print(
            f"flash-attn {flash_attn.__version__}: "
            "import and CUDA smoke test passed"
        )
        return 0
    except Exception as exc:
        print(f"flash-attn verification failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    module_parser = subparsers.add_parser("module-available")
    module_parser.add_argument("module")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--format", choices=("json", "shell"), default="json")
    subparsers.add_parser("verify-main")
    flash_environment_parser = subparsers.add_parser("flash-environment")
    flash_environment_parser.add_argument(
        "--format", choices=("json", "shell"), default="json"
    )
    flash_wheel_parser = subparsers.add_parser("find-flash-wheel")
    flash_wheel_parser.add_argument(
        "--format", choices=("json", "shell"), default="json"
    )
    subparsers.add_parser("test-flash-attention")
    args = parser.parse_args()

    if args.command == "module-available":
        return module_available(args.module)
    if args.command == "status":
        return show_status(args.format)
    if args.command == "verify-main":
        return verify_main()
    if args.command == "flash-environment":
        return show_flash_environment(args.format)
    if args.command == "find-flash-wheel":
        return find_flash_wheel(args.format)
    if args.command == "test-flash-attention":
        return test_flash_attention()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
