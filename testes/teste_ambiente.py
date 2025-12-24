from entidades.Ambiente import criar_ambiente

def test_criar():
    ambiente = criar_ambiente()
    assert ambiente.shape == (10, 10)