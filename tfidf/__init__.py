"""TF-IDF calculater."""
from typing import Callable, TypeVar, Generic, Collection, Optional, Dict
from collections import defaultdict
import math

Term = TypeVar('Term')
Doc = TypeVar('Doc')


__version__ = "0.0.1"


class TfIdf(Generic[Term, Doc]):
    def __init__(
        self,
        get_terms_of_doc: Callable[[Doc], Collection[Term]],
        get_docs: Callable[[], Collection[Doc]]
    ):
        self.get_terms_of_doc = get_terms_of_doc
        self.get_docs = get_docs
        self._idx_to_term: Optional[Dict[int, Term]] = None
        self._term_hash_to_idx: Optional[Dict[int, int]] = None
        self._idx_to_doc: Optional[Dict[int, Doc]] = None
        self._doc_hash_to_idx: Optional[Dict[int, int]] = None
        self._doc_to_term_to_term_count: Optional[Dict[int, Dict[int, int]]] = None
        self._term_to_doc_count: Optional[Dict[int, int]] = None
        self._term_to_idf: Optional[Dict[int, float]] = None
        self._doc_to_term_to_tf: Optional[Dict[int, Dict[float]]] = None
        self._doc_to_term_to_tfidf: Optional[Dict[int, Dict[float]]] = None
    @property
    def idx_to_term(self) -> Dict[int, Term]:
        if self._idx_to_term is None:
            terms = sorted(set(term for doc in self.idx_to_doc.values() for term in self.get_terms_of_doc(doc)))
            self._idx_to_term = {
                idx_term[0]: idx_term[1]
                for idx_term in enumerate(terms, start=1)
            }
        return self._idx_to_term
    @property
    def term_hash_to_idx(self) -> Dict[int, int]:
        if self._term_hash_to_idx is None:
            self._term_hash_to_idx = defaultdict(int)
            self._term_hash_to_idx.update({
                hash(idx_term[1]): idx_term[0]
                for idx_term in self.idx_to_term.items()
            })
        return self._term_hash_to_idx
    @property
    def idx_to_doc(self) -> Dict[int, Doc]:
        if self._idx_to_doc is None:
            self._idx_to_doc = {
                idx_doc[0]: idx_doc[1]
                for idx_doc in enumerate(self.get_docs(), start=0)
            }
        return self._idx_to_doc
    @property
    def doc_hash_to_idx(self) -> Dict[int, int]:
        if self._doc_hash_to_idx is None:
            self._doc_hash_to_idx = defaultdict(int)
            self._doc_hash_to_idx.update({
                hash(idx_doc[1]): idx_doc[0]
                for idx_doc in self.idx_to_doc.items()
            })
        return self._doc_hash_to_idx
    @property
    def doc_to_term_to_term_count(self) -> Dict[int, Dict[int, int]]:
        if self._doc_to_term_to_term_count is None:
            self._doc_to_term_to_term_count = defaultdict(lambda: defaultdict(int))
            for doc_idx, doc in self.idx_to_doc.items():
                _term_to_term_count = self._doc_to_term_to_term_count[doc_idx]
                for term_idx in map(lambda term: self.term_hash_to_idx[hash(term)], self.get_terms_of_doc(doc)):
                    _term_to_term_count[term_idx] += 1
        return self._doc_to_term_to_term_count
    @property
    def term_to_doc_count(self) -> Dict[int, int]:
        if self._term_to_doc_count is None:
            self._term_to_doc_count = defaultdict(int)
            for doc in self.idx_to_doc.values():
                for term_idx in set(map(lambda term: self.term_hash_to_idx[hash(term)], self.get_terms_of_doc(doc))):
                    self._term_to_doc_count[term_idx] += 1
        return self._term_to_doc_count
    @property
    def term_to_idf(self) -> Dict[int, float]:
        if self._term_to_idf is None:
            doc_total_count = math.log(len(self.idx_to_doc))
            self._term_to_idf = {
                term_idx: doc_total_count - math.log(doc_count)
                for term_idx, doc_count in self.term_to_doc_count.items()
            }
        return self._term_to_idf
    @property
    def doc_to_term_to_tf(self) -> Dict[int, Dict[int, float]]:
        if self._doc_to_term_to_tf is None:
            self._doc_to_term_to_tf = defaultdict(lambda: defaultdict(int))
            for doc_idx in self.idx_to_doc.keys():
                _term_to_tf = self._doc_to_term_to_tf[doc_idx]
                term_total_count = sum(self.doc_to_term_to_term_count[doc_idx].values())
                _term_to_tf.update({
                    term_idx_count[0]: term_idx_count[1] / term_total_count
                    for term_idx_count in self.doc_to_term_to_term_count[doc_idx].items()
                })
        return self._doc_to_term_to_tf
    @property
    def doc_to_term_to_tfidf(self) -> Dict[int, Dict[int, float]]:
        if self._doc_to_term_to_tfidf is None:
            self._doc_to_term_to_tfidf = defaultdict(lambda: defaultdict(int))
            for doc_idx in self.idx_to_doc.keys():
                _term_to_tfidf = self._doc_to_term_to_tfidf[doc_idx]
                _term_to_tfidf.update({
                    term_idx_tf[0]: term_idx_tf[1] * self.term_to_idf[term_idx_tf[0]]
                    for term_idx_tf in self.doc_to_term_to_tf[doc_idx].items()
                })
        return self._doc_to_term_to_tfidf
    def idx_of_term(self, term: Term) -> int:
        return self.term_hash_to_idx[hash(term)]
    def decode_term_idx(self, idx: int) -> Optional[Term]:
        return self.idx_to_term.get(idx)
    @property
    def idf(self) -> Dict[Term, float]:
        return {
            self.decode_term_idx(term_idx): idf
            for term_idx, idf in self.term_to_idf.items()
        }
    @property
    def tf(self) -> Dict[int, Dict[Term, float]]:
        return {
            doc: {
                self.decode_term_idx(term_idx): tf
                for term_idx, tf in term_to_tf.items()
            }
            for doc, term_to_tf in self.doc_to_term_to_tf.items()
        }
    @property
    def tfidf(self) -> Dict[int, Dict[Term, float]]:
        return {
            doc: {
                self.decode_term_idx(term_idx): tfidf
                for term_idx, tfidf in term_to_tfidf.items()
            }
            for doc, term_to_tfidf in self.doc_to_term_to_tfidf.items()
        }