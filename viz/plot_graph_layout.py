import networkx
import matplotlib
matplotlib.use("Qt5Agg")

graph_edge_hash = {('meme_gang','beaten_up'):'fight',\
                   ('meme_gang','escaped'):'escape',\
                   ('meme_gang','hidden'):'hide'}
G = networkx.DiGraph()
for k,v in graph_edge_hash.items():
    G.add_edge(k[0],k[1])

graph_pos = networkx.spring_layout(G)
networkx.draw_networkx_nodes(G,graph_pos,node_size=1600,alpha=1,node_color="white",edgecolors="black",linewidths=1)
networkx.draw_networkx_edges(G,graph_pos,width=1,alpha=0.3,edge_color='black')
networkx.draw_networkx_labels(G, graph_pos,font_size=12,font_family='sans-serif')
networkx.draw_networkx_edge_labels(G, graph_pos, edge_labels=graph_edge_hash, label_pos=0.3)
limits=matplotlib.pyplot.axis('off')
matplotlib.pyplot.show()