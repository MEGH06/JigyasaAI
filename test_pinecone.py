import traceback
try:
    import pinecone
    print("Pinecone imported okay")
except Exception as e:
    with open('out_utf8.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
