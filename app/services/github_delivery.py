class GitHubDelivery:
    def __init__(self, github_client):
        self.github = github_client

    def deliver(
        self,
        pull_request_number: int,
        markdown: str,
    ) -> dict:
        if pull_request_number is None:
            raise ValueError(
                "Cannot post PR comment because no PR number was found"
            )

        return self.github.post_issue_comment(
            issue_number=pull_request_number,
            body=markdown,
        )

    def deliver_commit_comment(
        self,
        sha: str,
        markdown: str,
    ) -> dict:
        if not sha.strip():
            raise ValueError(
                "Cannot post commit comment because commit sha is missing"
            )

        return self.github.post_commit_comment(
            sha=sha,
            body=markdown,
        )