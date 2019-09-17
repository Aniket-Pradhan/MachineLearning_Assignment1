import urllib.request, os
from tqdm import tqdm

class TqdmUpTo(tqdm):
	"""Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
	def update_to(self, b=1, bsize=1, tsize=None):
		if tsize is not None:
			self.total = tsize
		self.update(b * bsize - self.n)  # will also set self.n = b * bsize
