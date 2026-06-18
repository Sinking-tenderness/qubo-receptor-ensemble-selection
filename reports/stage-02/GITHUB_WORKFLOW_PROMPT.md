# GitHub Workflow Handoff for the Stage 2 Teaching Conversation

Give the following instruction to the conversation teaching Stage 2:

```text
From now on, use this existing research repository for all Stage 2 code and
documentation:

Local repository:
D:\量子×蛋白质\qubo-receptor-ensemble-selection

GitHub repository:
https://github.com/Sinking-tenderness/qubo-receptor-ensemble-selection

The repository has already been initialized and the main branch is published.
Do not create another repository and do not rewrite existing history.

At the start of each task:
1. Read README.md and the relevant directory documentation.
2. Run `git status -sb` and inspect existing changes.
3. Never delete or revert changes that I made unless I explicitly approve it.
4. Keep raw datasets, prepared ligand libraries, docking poses, MD trajectories,
   credentials, caches, and large generated output out of Git.

For each meaningful completed module:
1. Implement one small, coherent unit of work.
2. Validate it with a small example or automated test.
3. Explain the result and wait for me to complete any required learning exercise.
4. Update README, configuration, manifest, or stage report as appropriate.
5. Review `git status` and `git diff` before staging.
6. Stage only files belonging to the completed module.
7. Commit with a concise message using one of these patterns:
   - feat: add target and structure selection workflow
   - feat: add receptor preparation pipeline
   - test: validate crystal-ligand redocking
   - feat: add batch docking pipeline
   - feat: add virtual-screening metrics
   - docs: update stage 2 report
8. Push the commit to the tracked GitHub branch.
9. Report the commit hash, pushed branch, tests performed, and files excluded.

Do not commit after every tiny edit. Commit after a coherent module works and
has been checked. Do not upload secrets, access tokens, copyrighted datasets,
or files whose redistribution rights are unclear. If a large file is genuinely
needed, discuss a manifest, release asset, external data repository, or Git LFS
before adding it.

Use feature branches for experimental or risky work. Small verified teaching
modules may be committed to main while I am the sole contributor. Never use
force-push or destructive Git commands without my explicit approval.

If `git push` tries to use the unavailable proxy at 127.0.0.1:7897, run the push
for this command with proxy settings disabled rather than modifying unrelated
global settings:

git -c http.proxy= -c https.proxy= push
```

The Chinese workspace path may display differently across terminals. Always
confirm the resolved repository with `git rev-parse --show-toplevel` before
editing or committing.
