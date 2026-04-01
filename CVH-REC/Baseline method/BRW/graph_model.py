from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from random_walk_node2vec import generate_random_walks


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0

    def on_epoch_end(self, model):
        loss_cumulative = model.get_latest_training_loss()
        loss_now = loss_cumulative - self.loss_previous_step
        self.loss_previous_step = loss_cumulative
        print(f"Epoch #{self.epoch + 1} end, Loss this epoch: {loss_now}")
        self.epoch += 1


def generate_embeddings(G, dimensions=128, walk_length=20, num_walks=80, workers=4, phi=0.4, omega=0.6):
    walks = generate_random_walks(G, walk_length=walk_length, num_walks=num_walks, phi=phi, omega=omega)
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=5,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=5,
        compute_loss=True,
        callbacks=[EpochLogger()]
    )
    return model
