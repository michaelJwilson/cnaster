import pytest
import numpy as np
import anndata


@pytest.fixture
def mock_visium():
    gene_names = ["MIR1302-2HG", "FAM138A", "OR4F5", "AL627309.1", "AL627309.3"]
    barcodes = [
        "AAACAAGTATCTCCCA-1",
        "AAACAGAGCGACTCCT-1",
        "AAACGAGACGGTTGAT-1",
        "AAACGGGTTGGTATCC-1",
        "AAACTCGTGATATAAG-1",
    ]
    # Create a 5x5 matrix of counts
    X = np.arange(25).reshape(5, 5)
    adata = anndata.AnnData(X=X, obs={"barcode": barcodes}, var={"gene": gene_names})
    adata.obs_names = barcodes
    adata.var_names = gene_names
    return adata

def test_visium_fixture(mock_visium):
    assert list(mock_visium.var_names) == ['MIR1302-2HG', 'FAM138A', 'OR4F5', 'AL627309.1', 'AL627309.3']
    assert list(mock_visium.obs_names) == [
        'AAACAAGTATCTCCCA-1',
        'AAACAGAGCGACTCCT-1',
        'AAACGAGACGGTTGAT-1',
        'AAACGGGTTGGTATCC-1',
        'AAACTCGTGATATAAG-1'
    ]
