memory = []

def add(msg):
    memory.append(msg)
    
def get():
    return memory[-5:]