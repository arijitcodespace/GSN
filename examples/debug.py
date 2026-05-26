import h5py
import numpy as np
import tensorflow as tf
import sys, os

from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from gsn.datasets import load_tgb
from gsn.train.loop import GSNLinkPredictor

CKPT = './checkpoints/tgbl-wiki'
(train, val, test), meta = load_tgb('tgbl-wiki', root='data/')
model = GSNLinkPredictor.from_pretrained(CKPT, edge_feat_dim=int(meta['edge_feat_dim']))

# Ground truth from H5
with h5py.File(f'{CKPT}/best.weights.h5', 'r') as f:
    h5_state = f['blocks/persistent_gsn_block/state_table/vars/0'][()]

loaded_state = model.state_tables[0]._table.numpy()

print(f'H5    state[42, :4] = {h5_state[42, :4]}')
print(f'Model state[42, :4] = {loaded_state[42, :4]}')
print(f'Exact match: {np.allclose(h5_state, loaded_state)}')

# Also check a trainable weight to see if everything is shifted
with h5py.File(f'{CKPT}/best.weights.h5', 'r') as f:
    h5_emb = f['layers/embedding/vars/0'][()]
model_emb = model.node_id_emb.embeddings.numpy()
print(f'\nEmbedding match: {np.allclose(h5_emb, model_emb)}')