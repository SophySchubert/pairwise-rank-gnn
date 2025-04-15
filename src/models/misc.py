import torch
def create_document_mask(batch_size, pair_indices, max_num_nodes):
    """
    Erstellt eine Dokumentmaske für die Datenpaare in der Batch.

    Args:
        batch_size (int): Anzahl der Paare in der Batch.
        pair_indices (list of tuples): Liste der Indizes der Paare in der Batch.
        max_num_nodes (int): Maximale Anzahl von Knoten in einem Graphen.

    Returns:
        torch.Tensor: Die Dokumentmaske.
    """
    # Initialisiere die Maske mit Nullen
    document_mask = torch.zeros((batch_size, max_num_nodes, max_num_nodes), dtype=torch.bool)

    for i, (idx_a, idx_b) in enumerate(pair_indices):
        # Setze die Maske für das aktuelle Paar
        document_mask[i, idx_a, idx_b] = True
        document_mask[i, idx_b, idx_a] = True  # Optional, falls bidirektional

    return document_mask

if __name__ == '__main__':
    import torch
    from torch.nn.attention.flex_attention import flex_attention


    def create_document_mask(batch_size, pair_indices, max_num_nodes):
        """
        Erstellt eine Dokumentmaske für die Datenpaare in der Batch.

        Args:
            batch_size (int): Anzahl der Paare in der Batch.
            pair_indices (list of tuples): Liste der Indizes der Paare in der Batch.
            max_num_nodes (int): Maximale Anzahl von Knoten in einem Graphen.

        Returns:
            torch.Tensor: Die Dokumentmaske.
        """
        # Initialisiere die Maske mit Nullen
        document_mask = torch.zeros((batch_size, max_num_nodes, max_num_nodes), dtype=torch.bool)

        for i, (idx_a, idx_b) in enumerate(pair_indices):
            # Setze die Maske für das aktuelle Paar
            document_mask[i, idx_a, idx_b] = True
            document_mask[i, idx_b, idx_a] = True  # Optional, falls bidirektional

        return document_mask


    # Beispiel: Verwendung in einem Batch
    batch_size = 4
    max_num_nodes = 10
    pair_indices = [(0, 1), (2, 3), (4, 5), (6, 7)]  # Beispielhafte Paare

    # Erstelle die Maske
    document_mask = create_document_mask(batch_size, pair_indices, max_num_nodes)

    # Beispielhafte Eingabedaten für flex_attention
    query = torch.rand(batch_size, max_num_nodes, 32, 2)  # (Batch, Nodes, Features)
    key = torch.rand(batch_size, max_num_nodes, 32, 2)
    value = torch.rand(batch_size, max_num_nodes, 32, 2)
    print("Query shape:", query.shape)

    # Wende flex_attention an
    output = flex_attention(query, key, value, block_mask=document_mask)
    print(output.shape)