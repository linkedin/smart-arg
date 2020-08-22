# CONTRIBUTION

## Contribution Agreement

As a contributor, you represent that the code you submit is your original work or
that of your employer (in which case you represent you have the right to bind your
employer). By submitting code, you (and, if applicable, your employer) are
licensing the submitted code to LinkedIn and the open source community subject
to the BSD 2-Clause license.

## Responsible Disclosure of Security Vulnerabilities

**Do not file an issue on Github for security issues.**  Please review
the [guidelines for disclosure][disclosure_guidelines].  Reports should
be encrypted using PGP ([public key][pubkey]) and sent to
[security@linkedin.com][disclosure_email] preferably with the title
"Vulnerability in Github LinkedIn/smart-arg - &lt;short summary&gt;".

## Setup for Development

```shell-session
# # Uncomment the next a few lines to set up a virtual environment and install the packages as needed
# python3 -m venv .venv
# . .venv/bin/activate
# pip install -U setuptools pytest flake8 mypy

python3 setup.py develop

# Happily make changes
# ...

mypy
flake8 smart_arg.py
pytest
```


## Tips for Getting Your Pull Request Accepted

1. Make sure all new features are tested and the tests pass.
2. Bug fixes must include a test case demonstrating the error that it fixes.
3. Open an issue first and seek advice for your change before submitting
   a pull request. Large features which have never been discussed are
   unlikely to be accepted. **You have been warned.**

[disclosure_guidelines]: https://www.linkedin.com/help/linkedin/answer/62924
[pubkey]: https://www.linkedin.com/help/linkedin/answer/79676
[disclosure_email]: mailto:security@linkedin.com?subject=Vulnerability%20in%20Github%20LinkedIn/smart-arg%20-%20%3Csummary%3E
