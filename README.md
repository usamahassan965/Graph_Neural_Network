# Graph_Neural_Network
- A graph consists of nodes (or vertices) and connection between nodes (edges). Information about these connections in a graph can be represented in an adjacency matrix.
- Elements of adjacency matrix indicate the connected nodes with a '1' and disconnected nodes with a '0'.
- Besides the adjacency matrix , there also exists another way of representation which is adjacency lists.
- Further, the nodes can have node features and edges have their edge features.
- To understand graph, let's assume graph is a molecule and nodes are atoms and edges are bond type.
  
## Applications of Graph Neural Networks
1. Medicine or Pharmacy (related with molecular data)
2. E-commerce companies (to build recommender systems)
3. Social networks (facebook - people stand for nodes which are connected in certain relationship)
4. 3D games (graph structure is also find in 3d games where objects are modeled as a polygon mesh)
5. Machine Learning applications  (prediction of unlabelled nodes - Link prediction , Edge level predictions- whether there will be a connection between 2 nodes in a graph)

## Motivation for GNNs
- GNNs simply extend classical feed forward models . In the past, there were variety of other approaches to handle graphs such as hand crafted features as input for machine learning models.
- However graph data has some interesting properties that make it difficult to work with.
- We will find out how gns cope with all of these difficulties.
- As you might know neural networks typically expect a fixed size input. This brings us to the first difficulty of graph data
- The size and shape of a graph might change within a data set . This might also be true for other data types such as images but here you can simply resize pad or crop the images to the same size. Such operations are not defined on graph data if you have additional nodes or edges you cannot simply remove them.
- Therefore we need a method that can handle arbitrary input shapes another feature of graphs which is called isomorphism and graph theory says that two graphs that look different can still be structurally identical if you flip the image on the left you get an entirely new image if you flip the graph on the left however the only thing that changes is the order of the nodes.
- The algorithm that is supposed to handle graph data therefore needs to be permutation invariant.
- This is actually also the reason why you cannot directly use the adjacency matrix as input for the feed forward network as it is sensitive to changes in the node order
- Finally the structure of graphs is non-euclidean for images you have a clear grid that can be expressed by x and y coordinates.
- Graphs are dynamic structures that may lay differently in the space and distance metrics such as the euclidean distance are not clearly defined.
- For instance you cannot really say how close node a and b are of course you can add 3d coordinates but they do not incorporate the edge information between nodes this is also the reason why the machine learning area around graphs is called geometric deep learning .

## Graph Neural Networks 
- The fundamental idea of gnns is to learn a neural network's suitable representation of graph data. This is also called representation learning.
- Using all the information about the graph including the note features and the connections stored in an adjacency matrix the g n outputs new representations which are also called embeddings .
- For each of the nodes these node embeddings contain the structural as well as the feature information of the other nodes in the graph. This means each node knows something about the other nodes e.g., the connection to these nodes and its context in the graph.
- The embeddings can finally be used to perform predictions the way how you use them heavily depends on the machine learning problem you want to solve.
- For instance if you want to perform node level predictions you would simply use the node embedding of a specific unlabeled node to obtain a prediction.
-  Let's assume this example:  graph has four labeled nodes and one unlabeled node which is white then you would simply use the embedding vector of this node and predict the nodes label with it if you want to perform graph level predictions.
- However you would use all of the node embeddings, combine them in a certain way and get a representation of the whole graph.
-  Alternatively you can include pooling operations to iteratively compress the graph into a fixed size vector. This representation can then be used to run a prediction similar nodes , meaning nodes with similar features or in similar contexts will lead to similar node embeddings same way similar graphs will lead to similar graph embeddings.
- Using a gnn the size of the node embeddings is a hyper parameter and can differ from the initial node feature size.
- Let's assume the graph input is a molecule again and the atom feature vectors have a size of 50.Tthis means you have 50 properties such as the atom type or the number of protons available. - For each node then the embedding can for instance have a size of 128. However these embedding values cannot directly be interpreted as they are an artificial compound of the node and edge information within the graph.
-  Finally edge features can also be processed in the gnn and will be combined into these node embeddings.
- Within the graph neural network you have several so called message passing layers these are the core building blocks of gnns they are responsible for combining the node and edge information into the node embeddings.
