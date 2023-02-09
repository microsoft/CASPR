# Contributing

We're looking for your help to improve CASPR (bug fixes, new features, documentation, etc).

## Contribute a code change
* Start by reading the [CASPR Paper](https://arxiv.org/abs/2211.09174)
* If your change is non-trivial or introduces new public facing APIs (discussed in more detail below) please use the [feature request issue template](https://github.com/microsoft/CASPR/issues/new?template=feature_request.md) to discuss it with the team and get consensus on the basic design and direction first. For all other changes, you can directly create a pull request (PR) and we'll be happy to take a look.
* Make sure your PR adheres to the [PR Guidelines](./docs/PR_Guidelines.md) established by the team.
* If you're unsure about any of the above and want to contribute, you're welcome to start a discussion with the team.

## Process details

Please search the [issue tracker](https://github.com/microsoft/CASPR/issues) for a similar idea first: there may already be an issue you can contribute to.

1. **Create Issue**
To propose a new feature or API please start by filing a new issue in the [issue tracker](https://github.com/microsoft/CASPR/issues).
Include as much detail as you have. It's fine if it's not a complete design: just a summary and rationale is a good starting point.

2. **Discussion**
We'll keep the issue open for community discussion until it has been resolved or is deemed no longer relevant.
Note that if an issue isn't a high priority or has many open questions then it might stay open for a long time.

3. **Owner Review**
The CASPR team will review the proposal and either approve or close the issue based on whether it broadly aligns with the CASPR Roadmap and contribution guidelines.

4. **Implementation**
* A feature can be implemented by you, the CASPR team, or other community members.  Code contributions are greatly appreciated: feel free to work on any reviewed feature you proposed, or choose one in the backlog and send us a PR. If you are new to the project and want to work on an existing issue, we recommend starting with issues that are tagged with “good first issue”. Please let us know in the issue comments if you are actively working on implementing a feature so we can ensure it's assigned to you.
* Unit tests: New code *must* be accompanied by unit tests.
* Documentation and sample updates: If the PR affects any of the documentation or samples then include those updates in the same PR.
<!-- * Build instructions are [here](https://CASPR/build). -->
* Once a feature is complete and tested according to the contribution guidelines follow these steps:
  * Follow the [standard GitHub process to open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
  * Add reviewers who have context from the earlier discussion. If you can't find a reviewer, add 'microsoft/CASPR'.
* Note: After creating a pull request, you might not see a build getting triggered right away. One of the
CASPR team members can trigger the build for you.

## Licensing guidelines

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot should automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

## Code of conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Report a security issue

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).

<!-- credits: we are heavily inspired by documentation practices of our colleagues in the ONNX Runtime team -->