#!/bin/sh
if ! pip list | grep mypy > /dev/null; then
    pip install mypy
fi
find . -name "*.pyi" -exec rm -rf {} \;
stubgen -o . tfidf
ed tfidf/__init__.pyi << EOF
/get_terms_of_doc/
s/Any/Callable[[Doc], Collection[Term]]/
/get_docs/
s/Any/Callable[[], Collection[Doc]]
w
q
EOF
