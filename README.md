# Graph_Neural_Network
- A graph consists of nodes (or vertices) and connection between nodes (edges). Information about these connections in a graph can be represented in an adjacency matrix.
- Elements of adjacency matrix indicate the connected nodes with a '1' and disconnected nodes with a '0'.
- Besides the adjacency matrix , there also exists another way of representation which is adjacency lists.
- Further, the nodes can have node features and edges have their edge features.
- To understand graph, let's assume graph is a molecule and nodes are atoms and edges are bond type.

  ![Screenshot (318)](https://github.com/usamahassan965/Graph_Neural_Network/assets/96824810/5380e079-855e-4eeb-b71a-13f21a892c33)

  
## Applications of Graph Neural Networks
1. Medicine or Pharmacy (related with molecular data)
2. E-commerce companies (to build recommender systems)
3. Social networks (facebook - people stand for nodes which are connected in certain relationship)
4. 3D games (graph structure is also find in 3d games where objects are modeled as a polygon mesh)
5. Machine Learning applications  (prediction of unlabelled nodes - Link prediction , Edge level predictions- whether there will be a connection between 2 nodes in a graph)

![Screenshot (320)](https://github.com/usamahassan965/Graph_Neural_Network/assets/96824810/92e2af92-06b8-4a4f-843d-5aca529172c2)

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

![Screenshot (321)](https://github.com/usamahassan965/Graph_Neural_Network/assets/96824810/9bf1cb8d-0577-4860-83b9-89e7dae0f993)

## Graph Neural Networks 
- The fundamental idea of gnns is to learn a neural network's suitable representation of graph data. This is also called representation learning.
- Using all the information about the graph including the node features and the connections stored in an adjacency matrix the gnn outputs representations which are also called embeddings .
- For each of the nodes, these node embeddings contain the structural as well as the feature information of the other nodes in the graph. This means each node knows something about the other nodes e.g., the connection to these nodes and its context in the graph.
- The embeddings can finally be used to perform predictions the way how you use them heavily depends on the machine learning problem you want to solve.
- For instance if you want to perform node level predictions you would simply use the node embedding of a specific unlabeled node to obtain a prediction.
-  Let's assume this example:  graph has four labeled nodes and one unlabeled node which is white then you would simply use the embedding vector of this node and predict the nodes label with it
-   If you want to perform graph level predictions however you would use all of the node embeddings, combine them in a certain way and get a representation of the whole graph.
-  Alternatively you can include pooling operations to iteratively compress the graph into a fixed size vector. This representation can then be used to run a prediction.
-  Similar nodes , meaning nodes with similar features or in similar contexts, will lead to similar node embeddings. The same way similar graphs will lead to similar graph embeddings.
- Using a gnn the size of the node embeddings is a hyper parameter and can differ from the initial node feature size.
- Let's assume the graph input is a molecule again and the atom feature vectors have a size of 50.Tthis means you have 50 properties such as the atom type or the number of protons available. - For each node then the embedding can for instance have a size of 128. However these embedding values cannot directly be interpreted as they are an artificial compound of the node and edge information within the graph.
-  Finally edge features can also be processed in the gnn and will be combined into these node embeddings.
- Within the graph neural network you have several so called message passing layers these are the core building blocks of gnns they are responsible for combining the node and edge information into the node embeddings.

![Screenshot (324)](https://github.com/usamahassan965/Graph_Neural_Network/assets/96824810/c1de4260-91c6-493b-ad18-66bb5258a91f)

## Understanding Graph Neural Networks
- The graph information, including node features and structural properties, is fed through message passing layers. These layers construct the node embeddings that contain the knowledge about other nodes and edges in a compressed format.

## Message Passing Layers
 - This is done by gathering the current node information of neighbor nodes, combining it in certain ways to get a new embedding and updating that node features or states with these embeddings. This approach is also called graph convolution and it can be seen as an extension of convolutions to graph data.
 - let's have a look at this visualization from a paper which shows image convolutions on the left and graph convolutions on the right.
 - For images you simply slide learnable kernels over the regular grid structure of the pixels which extracts the most important information.
 - This can also be seen as combining the information in a local area by using all the pixels in this neighborhood.
 - For non-euclidean graph structures, this idea is extended as we simply use the information in a node's neighborhood and combine it into a new embedding vector.
 -  If we look at the red node in a graph, this simply means that the neighboring nodes share their current embeddings with it. This is done simultaneously for all nodes. This sharing mechanism is also called message passing as the states can be seen as many messages passing back and forth between the notes.
 - Now let's have a look at a concrete example to better understand this idea .let's assume this is our input graph in the following: we see that there is one yellow node (with the number 1 with a yellow node feature vector or state).
 - This is the node we will focus on in the following to update the node state. We collect the information of the direct neighbors which means we perform the message passing .
 - What we end up with, is the information about our current node state and the information about our neighbors node states. These states are usually denoted with h.
 - Currently we are in time step k then we perform an aggregation on the neighbor states to combine their information.
 - Finally, we put our current state and the combined neighbor information together to get a new state or embedding in layer k plus 1.
 - Note how some of the feature information of the blue nodes enters the state of the yellow node. Now we update our annotations in the graph. This message passing is done by all nodes and therefore we have new embeddings for every node in our graph.
 - The size of these new embeddings is a hyper parameter and depends on the graph data you use.
 - As you can see the node with the number five only holds information about the blue node and itself because it's green and blue. Currently this node doesn't know about our yellow node with the number one but this will change.
 - Let's perform another message passing step to see what happens and actually we can perform several of these message passing steps which corresponds to the number of layers in the gnn.
 - Again we use the current node embedding of our yellow node , collect the state messages of its neighbors and aggregate them in some way. If we update the yellow nodes embedding now we can see that some information about the green node passed into it.
 -  This means that node 1 knows something about node 5 but additionally in our example every single node in the graph knows something about all other nodes.
 - This knowledge is stored in each of our node embeddings and contains the feature based as well as structural information about the nodes.
 - Eventually we can use the embeddings to perform predictions as they contain all the information about the graph that we need and this is the basic idea of gnns.
 - We learn these embeddings by iteratively combining the node information in a local neighborhood. Iteratively means we first learn something about the direct neighbors then about the neighbor's neighbors and so on this local feature aggregation can be compared to learnable cnn kernels.

## GNN Depth / Computation Graph
- We can actually visualize how deep we dive into the graph from each node's perspective. This means that we can understand which neighbors and neighbor neighbors and so on we learn about this is usually called the computation graph.
- For a specific node if we restructure our graph like this we can automatically see which nodes are the direct neighbors of our yellow node. This means, in the first layer of our message passing gnn, the yellow node incorporates information about the blue nodes. Now if we add the direct neighbors for each of these nodes we can see the next layer.
- Two of our blue nodes are connected to blue and yellow and the third blue node is connected to green. We can see that after two layers, the yellow node already contains the information about all nodes in the graph.
- The number of layers in a gnn defines how many neighborhood hops we perform this number is a hyperparameter and depends on the graph data we use.
- If you have small graphs such as smaller molecules you can quickly learn all the information after only a few layers. The number of layers also depends on the learning task. Sometimes only a local area of the graph might be relevant for your predictions but stacking too many message passing layers in a gnn can also lead to a phenomenon called over smoothing.
- As you see in our previous example, the node embeddings contain most of the information already after the second layer. Hence if you keep combining these states over many more layers you will not learn anything new but instead make all node states indistinguishable from each other.
- There already exist methods such as paranorm which can handle these issues but for now let's assume we have no over smoothing in our gnn.

## Formal Definition of GNNs
- let's formulate the operations in the message passing layers more mathematically.
- The state update for a node u is mainly performed using the two already introduced operations: aggregate and update aggregate uses the states of all direct neighbors v of a node u and aggregates them in a specific way.
- Then the update operation uses the current state in time step k and combines it with the aggregated neighbor states. If we think of our previous example, our node u is the yellow node and its neighbors are the three blue nodes. We use their states in time step k and combine them with the yellow state to get a new embedding for the yellow node.

## Variants of GNNs
- Note that the basic formul stays the same for all variants of message passing graph neural networks. The only thing in which they are different is how they perform the update and aggregate functions.
-  Many different operations have already been published in literature and besides simple mean or max operations there are more advanced methods like recurrent neural networks.
- Let's go over some examples. One of the first famous works uses two interesting ideas. First of all, they aggregate the neighbor information as a normalized sum of the states.Additionally, they incorporate the update operation into this aggregation by adding a self loop for a particular node including it into the summation.
- This means update and aggregate are combined into one computation.
- Another work uses multi-layer perceptrons so basically feed forward networks to perform the aggregate operation. This means that there are learnable weights which can be optimized for the best aggregation of the neighbor states.
- Another popular paper applied the attention mechanism to gnns, this means that the importance of the features of the neighbor states is considered for the aggregation. As a result, the updated embedding contains more information about important neighbor features. Finally, gated graph neural networks use a recurrent unit to update the state iteratively over time.
- Besides these introduced variants they exist many more in the literature but don't be scared they all just use different approaches for the aggregate and update function.
