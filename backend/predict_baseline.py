from collections import defaultdict, Counter
import re

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.counts = defaultdict(Counter)

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'([.,!?;:()"])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return []
        return text.split()

    def train_from_lines(self, lines):
        for line in lines:
            tokens = self._tokenize(line)
            if not tokens:
                continue
            tokens = ['<s>'] + tokens + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                nxt = tokens[i + self.n - 1]
                self.counts[context][nxt] += 1

    def predict(self, context_text, k=5):
        tokens = self._tokenize(context_text)
        if len(tokens) == 0:
            ctx = ('<s>',) * (self.n - 1)
        else:
            ctx_tokens = (['<s>'] * max(0, (self.n - 1) - len(tokens)) +
                          tokens[-(self.n - 1):])
            ctx = tuple(ctx_tokens)
        candidates = self.counts.get(ctx, None)
        if not candidates:
            if len(ctx) > 1:
                short_ctx = ctx[1:]
                candidates = self.counts.get(short_ctx, None)
            if not candidates:
                return []
        results = []
        for token, _ in candidates.most_common(k * 3):
            if token == '</s>':
                continue
            results.append(token)
            if len(results) >= k:
                break
        return results
