import keras

class DataNode:
    def __init__(
        the_node,
        data,
        parents='default',
        children='default',
    ): # parents and children as None indicate a terminason node when walking it
        # can be an empty set, usually means the node is not linked yet
        # can have multiple children or parents potentially but not implemented yet
        if parents == 'default':
            parents = set()
        if children == 'default':
            children = set()
        if parents is not None:
            assert isinstance(parents,set) and 0 <= len(parents)
        if children is not None:
            assert isinstance(children,set) and 0 <= len(children)
        the_node.data = data
        the_node.parents = parents
        the_node.children = children
        the_node.inducted_layer = None

    def get_adopted(the_node, new_parent):
        assert the_node.parents is None, the_node.inspect()
        the_node.parents = set([new_parent])
        if new_parent.children is None:
            new_parent.children = set()
        new_parent.children.add(the_node)
        assert new_parent in the_node.parents
        assert the_node in new_parent.children
        return the_node

    def add_parent(the_node, new_parent):
        assert the_node.parents is not None, the_node.inspect()
        the_node.parents.add(new_parent)
        new_parent.children.add(the_node)
        assert new_parent in the_node.parents
        assert the_node in new_parent.children
        return the_node

    def remove_parent(the_node, former_parent):
        assert the_node.parents, the_node.inspect()
        assert former_parent in the_node.parents, the_node.inspect()
        the_node.parents.remove(former_parent)
        former_parent.children.remove(the_node)
        return the_node

    def swap_parent(the_node, former_parent_and_new_parent ):
        former_parent, new_parent = former_parent_and_new_parent
        assert the_node.parents, the_node.inspect()
        the_node.remove_parent(former_parent)
        the_node.add_parent(new_parent)
        assert former_parent not in the_node.parents
        assert the_node not in former_parent.children
        assert new_parent in the_node.parents
        assert the_node in new_parent.children
        return the_node

    def intercalate(the_node, new_parent_and_new_child):
        new_parent, new_child = new_parent_and_new_child
        assert new_child in new_parent.children
        assert the_node.is_displaced(), the_node.inspect()
        assert new_child in new_parent.children and new_parent in new_child.parents, the_node.inspect()
        new_child.swap_parent( (new_parent, the_node) )
        the_node.add_parent(new_parent)
        assert new_child in the_node.children
        assert the_node in new_parent.children
        return the_node

    def substitute(the_node, substituee_node):
        for child in substituee_node.children:
            child.parents.remove(substituee_node)
            child.parents.add(the_node)
        for parent in substituee_node.parents:
            parent.children.remove(substituee_node)
            parent.children.add(the_node)
        the_node.parents, substituee_node.parents = substituee_node.parents, the_node.children
        the_node.children, substituee_node.children = substituee_node.children, the_node.children
        return the_node
        
    def extricate(the_node, former_parent_and_former_child):
        former_parent, former_child = former_parent_and_former_child
        assert the_node.is_placed(), the_node.inspect()
        
        #remove connection between parent and node
        former_parent.children.remove(the_node)
        the_node.parents.remove(former_parent)
        #remove connection between node and child
        the_node.children.remove(former_child)
        former_child.parents.remove(the_node)
        # create new connection between former parent and former child
        former_parent.children.add(former_child)
        former_child.parents.add(former_parent)
        
        # former_child.swap_parent(former_parent_and_new_parent=(the_node, former_parent))
        assert the_node not in former_parent.children
        assert former_child not in the_node.children
        return the_node
        
    def displace(the_node):
        assert the_node.is_placed(), the_node.inspect()
        former_children = the_node.children
        former_parents = the_node.parents
        for former_parent in former_parents:    
            for former_child in former_children:
                the_node.extricate((former_parent , former_child))
        return the_node
        
    def is_placed(the_node):
        return the_node.parents and the_node.children

    def is_displaced(the_node):
        return not the_node.is_placed()
    
    def annilate(the_node):
        assert the_node.is_displaced(), the_node.inspect()
        del the_node

    def walk_down(the_node):
        yield the_node
        if the_node.children:
            for child in the_node.children:
                yield from child.walk_down()

    def walk_up(the_node):
        yield the_node
        if the_node.parents:
            for parent in the_node.parents:
                yield from parent.walk_up()
        
    def inspect(the_node):
        return f"{the_node = }{the_node.parents = } {the_node.children = }"

    def lineage(the_node, last_nodes=set(), excludee_nodes=set()):
        lineage = set()
        to_dos = set()
        looping_node = the_node
        # counter = 0
        if not isinstance(last_nodes,set):
            last_nodes=set([last_nodes])
        if not isinstance(excludee_nodes,set):
            excludee_nodes=set([excludee_nodes])
        while True:
            children_to_check = looping_node.children - excludee_nodes if looping_node.children else set()
            if children_to_check and looping_node not in last_nodes:
                for child in looping_node.children - excludee_nodes: #add all children to todo list
                    if child not in excludee_nodes:
                        lineage.add(child)
                        to_dos.add(child)
            if to_dos:
                looping_node = to_dos.pop() ## next in the list
            if not to_dos and not children_to_check: # we are done
                assert(not the_node in lineage)
                return lineage

    def ancestors(the_node, last_nodes=None, excludee_nodes=set()):
        ancestors = set()
        to_dos = set()
        looping_node = the_node
        if not isinstance(last_nodes,set):
            last_nodes=set([last_nodes])
        if not isinstance(excludee_nodes,set):
            excludee_nodes=set([excludee_nodes])
        while True:
            parents_to_check = looping_node.parents - excludee_nodes if looping_node.parents else set()
            if parents_to_check and looping_node not in last_nodes:
                for parent in looping_node.parents - excludee_nodes: #add all parents to todo list
                    if parent not in excludee_nodes:
                        ancestors.add(parent)
                        to_dos.add(parent)
            if to_dos:
                looping_node = to_dos.pop() ## next in the list
            if not to_dos and not parents_to_check: # we are done
                assert(the_node not in ancestors)
                return ancestors

    def clone(the_node, harbinger_node=None, ):
        return DataNode(
            data=the_node.data.clone(),
            parents=set(parent.clone(the_node) for parent in parents if parent is not harbinger_node),
            children=set(child.clone(the_node) for child in children if child is not harbinger_node),
        )
    
    def __str__(the_node):
        return f"node<{the_node.data}>"
    def __repr__(the_node):
        return the_node.__str__()


def path_data_graph_from_datas(datas):
    first_data_node = DataNode(data=datas[0] ,parents=None)
    last_data_node = first_data_node
    for data in datas[1:-1]:
        last_data_node = DataNode(data=data).add_parent(last_data_node)
    last_data_node = DataNode(data=datas[-1], children=None).add_parent(last_data_node)
    return first_data_node, last_data_node

def path_keras_model_from_last_data_node(
    last_data_node, 
    input_shape, 
    first_data_node=None,
): # -> keras model
    def inducted_layer_and_model_input_from_node(a_data_node, input_shape, first_data_node=None):
        layer = a_data_node.data.keras_layer()
        
        if a_data_node.parents is None or a_data_node is first_data_node:
            layer_input = keras.Input(input_shape)
            inducted_layer = layer(layer_input)
            model_input = layer_input 
            
        if a_data_node.parents:
            assert len(a_data_node.parents) == 1, f"only path graph modelis supproted {node.parents = }"
            parent_node = list(a_data_node.parents)[0]
            parent_inducted_layer, model_input = inducted_layer_and_model_input_from_node(
                parent_node, input_shape
            ) 
            inducted_layer = layer(parent_inducted_layer)

        return inducted_layer, model_input
        
    model_output, model_input = inducted_layer_and_model_input_from_node(last_data_node, input_shape, first_data_node=first_data_node)
    return keras.Model(inputs=model_input, outputs=model_output)
        
if __name__ == "__main__":
    def test():
        ip_data_node = DataNode(
            data=ConvolutionData(),
            parents=None,
        )
        op_data_node = DataNode(
            data=FlattenAndDenseData(),
            children=None,
        )
        op_data_node.add_parent(ip_data_node)
        keras_model = path_keras_model_from_last_data_node(op_data_node,input_shape=(48,48,1))
        print(f"{keras_model= }")
    test()
    def test():
        op_data_node = DataNode(
            data=FlattenAndDenseData(),
            parents=None,
            children=None,
        )
        keras_model = path_keras_model_from_last_data_node(op_data_node,input_shape=(48,48,1))
        print(f"{keras_model= }")
    test()
# complexe branched graphs are not supported yet
    # merging of branches (many children node) or splitting (many parents nodes)
        # need to specify (input node, input shape) couple(s) upfront


def graph_edges_from_top_node(top_node):
    def children_edges(a_node):
        if a_node.children:
            return list(
                (a_node, child) for child in a_node.children
            )
        return []
    return list(
        edge 
        for a_node in top_node.walk_down() 
        for edge in children_edges(a_node)
    )    

# import networkx
from matplotlib import pyplot

# def draw_edges(edges):
#     graph = networkx.DiGraph()
#     graph.add_edges_from(edges)
#     graph.nodes()
#     networkx.draw(graph,with_labels=True)

# def draw_from_top_data_node(top_node):
    # return draw_edges(graph_edges_from_top_node(top_node))


