
import keras
import pandas
import numpy

1/0 need rework

def generate_reversible_convolution_intercalating_from_parent_node( parent_and_child_data_nodes ):
    new_convolution_data_node = DataNode(data=ConvolutionData())
    def effect(parent_and_child_data_nodes): 
        new_convolution_data_node.intercalate( parent_and_child_data_nodes )
    def reverse(parent_and_child_data_nodes):
        new_convolution_data_node.extricate( parent_and_child_data_nodes )
    return effect, reverse

import random

def get_one_random_parent_data_node_from_graph(top_and_bot_node):
    top_node, bot_node = top_and_bot_node
    assert top_node.children, f"top node with no children ? {a_parent}"
    candidate_parent_nodes = top_node.lineage(excludee_nodes={bot_node})
    for a_parent in candidate_parent_nodes:
        assert a_parent.children is not None, f"parent with no children ? {a_parent}"
    candidate_parent_nodes.add(top_node)
    picked_parent_node = random.choice(list(candidate_parent_nodes))
    assert picked_parent_node.children, f"picked parent node has n children ! {picked_parent_node.inspect()}"
    return picked_parent_node

def get_one_random_child_data_node_from_parent_data_node(parent_node):
    assert parent_node.children, f"node has no children {parent_node = }"
    return random.choice(list(parent_node.children))

def generate_random_reversible_intercalating_from_graph( top_and_bot_node ):
    parent_node = get_one_random_parent_data_node_from_graph(top_and_bot_node)
    assert parent_node.children, f"node has no children {parent_node = }"
    child_node = get_one_random_child_data_node_from_parent_data_node(parent_node)
    effect, reverse = generate_reversible_convolution_intercalating_from_parent_node( (parent_node, child_node) )
    return  (parent_node, child_node) , effect, reverse 

if __name__ == "__main__":
    def test():
        input_data_node = DataNode(
            data=InputData((48,48,1)),
            parents=None,
        )
        convolution_data_node = None
        convolution_data_node = DataNode(
            data=ConvolutionData(shape=(3,2), features=4),
        )    
        output_data_node = DataNode(
            data=DenseData(7),
            children=None,
        )
        convolution_data_node.add_parent(input_data_node)
        output_data_node.add_parent(convolution_data_node)
    
        for _ in range(5):
            effectee, effect, _reverse = generate_random_reversible_intercalating_from_graph( (input_data_node, output_data_node) )
            effect(effectee)

        draw_from_top_data_node(input_data_node)

    test()

def get_all_convolution_data_nodes( top_and_bot_node ):
    top_node, bot_node = top_and_bot_node
    middle_nodes = top_node.lineage(excludee_nodes={bot_node})
    middle_convolution_nodes = list(
        middle_node for middle_node in middle_nodes if isinstance(middle_node.data, ConvolutionData)
    )
    return middle_convolution_nodes
    
def shape(convolution_data_node):
    convolution_data_node.data = convolution_data_node.data.widened_kernel()
    return convolution_data_node

def shape(convolution_data_node):
    convolution_data_node.data = convolution_data_node.data.unwidened_kernel()
    return convolution_data_node

def shape(convolution_data_node):
    convolution_data_node.data = convolution_data_node.data.heightened_kernel()
    return convolution_data_node

def shape(convolution_data_node):
    convolution_data_node.data = convolution_data_node.data.unheightened_kernel()
    return convolution_data_node

def augment_convolution_data_node_features(convolution_data_node):
    convolution_data_node.data = convolution_data_node.data.augmented_features()
    return convolution_data_node
    
def unaugment_convolution_data_node_features(convolution_data_node):
    convolution_data_node.data = convolution_data_node.data.unaugmented_features()
    return convolution_data_node

# randomized transformative graph operations
import random

def get_one_random_convolution_data_node_from_graph(top_and_bot_node):
    all_convolution_data_nodes = get_all_convolution_data_nodes(top_and_bot_node)
    assert all_convolution_data_nodes, f"not one conv node in the graph ? {all_convolution_data_nodes = }"
    return random.choice(list(all_convolution_data_nodes))

def generate_random_reversible_convolution_kernel_widening_from_graph( top_and_bot_node ):
    random_convolution_node = get_one_random_convolution_data_node_from_graph(top_and_bot_node)
    return (
        random_convolution_node,
        shape,
        shape,
    )
def generate_random_reversible_convolution_kernel_heightening_from_graph( top_and_bot_node ):
    random_convolution_node = get_one_random_convolution_data_node_from_graph(top_and_bot_node)
    return (
        random_convolution_node,
        shape,
        shape,
    )

def generate_random_reversible_convolution_feature_augmentation_from_graph( top_and_bot_node ):
    random_convolution_node = get_one_random_convolution_data_node_from_graph(top_and_bot_node)
    return (
        random_convolution_node,
        augment_convolution_data_node_features,
        unaugment_convolution_data_node_features,
    )

def generate_random_reversible_convolution_augmentation_from_graph( top_and_bot_node ):
    convolution_data_node = get_one_random_convolution_data_node_from_graph(top_and_bot_node)
    return random.choice([
        generate_random_reversible_convolution_kernel_widening_from_graph,
        generate_random_reversible_convolution_kernel_heightening_from_graph,
        generate_random_reversible_convolution_feature_augmentation_from_graph,
    ])(top_and_bot_node)

if __name__ == "__main__":
    def test():
        input_data_node = DataNode(
            data=InputData((48,48,1)),
            parents=None,
        )
        convolution_data_node = None
        convolution_data_node = DataNode(
            data=ConvolutionData(shape=(3,2), features=4),
        )    
        output_data_node = DataNode(
            data=DenseData(7),
            children=None,
        )
        convolution_data_node.add_parent(input_data_node)
        output_data_node.add_parent(convolution_data_node)
    
        for _ in range(5):
            effectee, effect, _reverse = generate_random_reversible_convolution_augmentation_from_graph( (input_data_node, output_data_node) )
            effect(effectee)

        draw_from_top_data_node(input_data_node)

    test()

import random
def generate_random_reversible_mutation_from_graph(data_graph):
    mutation = random.choices(
        [
            generate_random_reversible_convolution_augmentation_from_graph,
            generate_random_reversible_intercalating_from_graph,
        ],
        [3,1],
        k=1,
    )[0]
    return mutation(data_graph)

if __name__ == "__main__":
    def test():
        input_data_node = DataNode(
            data=InputData((48,48,1)),
            parents=None,
        )
        convolution_data_node = None
        convolution_data_node = DataNode(
            data=ConvolutionData(shape=(1,1), features=4),
        )    
        flatten_data_node = DataNode(
            data=FlattenData(),
        )
        output_data_node = DataNode(
            data=DenseData(units=7),
            children=None,
        )
        convolution_data_node.add_parent(input_data_node)
        flatten_data_node.add_parent(convolution_data_node)
        output_data_node.add_parent(flatten_data_node)

        # for _ in range(2):
        #     additive_effectee, additive_effect, _reverse = generate_random_reversible_intercalating_from_graph( (input_data_node, flatten_data_node) )
        #     additive_effect(additive_effectee)
        for _ in range(20):
            trasformative_effectee, transformative_effect, reverse = generate_random_reversible_convolution_augmentation_from_graph( (input_data_node, output_data_node) )
            transformative_effect(trasformative_effectee)
            reverse(trasformative_effectee)
        
        draw_from_top_data_node(input_data_node)

        data_graph = input_data_node, output_data_node

        return data_graph

    data_graph = test()



