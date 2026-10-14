# Publish to the existing GitHub repository

After downloading and extracting this package on Windows:

1. Open PowerShell in the extracted `parallel-sgd-matrix-factorization` folder.
2. Confirm Git is installed:

```powershell
git --version
```

3. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\publish_to_github.ps1
```

The script clones the existing repository, copies this cleaned public-ready version over it, creates one commit, and pushes to `main`.

If GitHub asks for authentication, complete the normal browser/device authentication. The script contains no credentials.

**Do not add the unpublished A-ASGD source before publication/IP approval.**
