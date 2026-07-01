# Making a release

> [!NOTE]
> VERSION needs to be formatted following the `v{major}.{minor}.{patch}` convention. We need to follow this convention to be able to retrieve versioned scripts.

## Major/Minor Release

### 1. Ensure your local repository is up to date with the upstream repository

```bash
git checkout main
git pull origin main
```

> [!WARNING]
> Do not merge other pull requests into `main` until the release is done. This is to ensure that the release is stable and does not include any untested changes. Announce internally (#trl-internal) to other maintainers that you are doing a release and that they must not merge PRs until the release is done.

### 2. Create a release branch from main

```bash
git checkout -b release-v{major}.{minor}
```

### 3. Change the version in the following files

- `.github/workflows/tests_latest.yml`:
  
  ```diff
  - with: { ref: v{major}.{minor-1}-release }
  + with: { ref: v{major}.{minor}-release }
  ```

- `CITATION.cff`

  ```diff
  - version: '{major}.{minor-1}'
  + version: '{major}.{minor}'
  ```

- `VERSION`

  ```diff
  - {major}.{minor}.0.dev0
  + {major}.{minor}.0
  ```

### 4. Commit and push these changes

```shell
git add .github/workflows/tests_latest.yml CITATION.cff VERSION
git commit -m 'Release: {major}.{minor}'
git push origin release-v{major}.{minor}
```

### 5. Create a pull request

from `release-v{major}.{minor}` to `main`, named `Release: v{major}.{minor}`, wait for tests to pass, and request a review.

### 6. Once the pull request is approved, merge it into `main`

It will automatically publish the new version of the package on PyPI.

### 7. Add a tag in git to mark the release

```shell
git checkout main
git pull origin main
git tag -a v{major}.{minor}.0 -m 'Adds tag v{major}.{minor}.0 for PyPI'
git push origin v{major}.{minor}.0
```

### 8. Create a branch `v{major}.{minor}-release` for future patch releases

```shell
git checkout -b v{major}.{minor}-release
git push origin v{major}.{minor}-release
```

This ensures that future patch releases (`v{major}.{minor}.1`, `v{major}.{minor}.2`, etc.) can be made separately from `main`.

### 9. Create a GitHub Release

1. Go to the repo’s [releases section](https://github.com/huggingface/trl/releases) on GitHub.
2. Click **Draft a new release**.
3. Select the `v{major}.{minor}.0` tag you just created in step 7.
4. Add a title (`v{major}.{minor}.0`) and a short description of what’s new.
5. Click **Publish Release**.

### 10. Bump to dev version

1. Create a branch `bump-dev-version-{major}.{minor+1}` from `main` and checkout to it.

  ```shell
  git checkout -b bump-dev-version-{major}.{minor+1}
  ```

2. Change the version in file `VERSION`:

  ```diff
  - {major}.{minor}.0
  + {major}.{minor+1}.0.dev0
  ```

3. Commit and push these changes

  ```shell
  git add VERSION
  git commit -m '⬆️ Bump dev version'
  git push origin bump-dev-version-{major}.{minor+1}
  ```

4. Create a pull request from `bump-dev-version-{major}.{minor+1}` to `main`, named `⬆️ Bump dev version`, and request urgent review.

5. Once the pull request is approved, merge it into `main`.

6. The codebase is now ready for the next development cycle, inform the team in the #trl-internal channel.

## Making a patch release

### 1. Ensure your local repository is up to date with the upstream repository

```bash
git checkout v{major}.{minor}-release
git pull origin main
```

### 2. Cherry-pick the changes you want to include in the patch release

```bash
git cherry-pick <commit-hash-0>
git cherry-pick <commit-hash-1>
...
```

### 3. Change the version in the file `VERSION`

```diff
- {major}.{minor}.{patch-1}
+ {major}.{minor}.{patch}
```

### 4. Commit and push these changes

```shell
git add VERSION
git commit -m 'Release: {major}.{minor}.{patch}'
git push origin v{major}.{minor}-release
```

### 5. Wait for the CI to pass

The CI will automatically publish the new version of the package on PyPI.

### 6. Add a tag in git to mark the release

```shell
git tag -a v{major}.{minor}.{patch} -m 'Adds tag v{major}.{minor}.{patch} for PyPI'
git push origin v{major}.{minor}.{patch}
```

#### 7. Create a GitHub Release

1. Go to the repo’s [releases section](https://github.com/huggingface/trl/releases) on GitHub.
2. Click **Draft a new release**.
3. Select the `v{major}.{minor}.{patch}` tag you just created in step 7.
4. Add a title (`v{major}.{minor}.{patch}`) and a short description of what’s new.
5. Click **Publish Release**.
