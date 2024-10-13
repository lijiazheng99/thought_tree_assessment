from anytree import NodeMixin, RenderTree
from anytree.exporter import JsonExporter, DotExporter
import uuid
import json

def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        if node.rationale is not None:
            print("%s%s (%s) - %s" % (pre, node.foo, node.weight, node.rationale))
        else:
            print("%s%s (%s)" % (pre, node.foo, node.weight))

def save_graph(tree, path):
    DotExporter(tree, nodeattrfunc=lambda node: 'label="{}"'.format(node.foo), edgeattrfunc=lambda parent, child: "style=bold,label=%0.2f" % (child.weight)).to_picture(path)

def find_paths(node, path=[]):
    if node.weight is not None:
        path.append({node.foo: node.weight, "rationale": node.rationale})
    paths = []
    if not node.children:
        return [path]
    for child in node.children:
        paths.extend(find_paths(child, path.copy()))
    return paths

def find_paths(node, path=[]):
    # print(node.foo, node.weight, node.rationale)
    if node.weight is not None:
        path.append({node.foo: node.weight, "rationale": node.rationale})
    paths = []
    if not node.children:
        return [path]
    for child in node.children:
        paths.extend(find_paths(child, path.copy()))
    return paths

def remove_llm(tree):
    if tree.foo.startswith("LLM"):
        tree.parent = None
    else:
        for child in tree.children:
            remove_llm(child)

def add_tree(parent_node, json_tree):
    tree = WNode(json_tree["foo"], rationale = json_tree["rationale"], weight=json_tree["weight"], parent=parent_node)
    if "children" in json_tree:
        for child in json_tree["children"]:
            add_tree(tree, child)
    else:
        pass

def importer(textual_data):
    json_tree = json.loads(textual_data)
    tree = WNode(json_tree["foo"], rationale = json_tree["rationale"], weight=json_tree["weight"])
    for child in json_tree["children"]:
        add_tree(tree, child)
    return tree

class WNode(NodeMixin):
    def __init__(self, foo, parent=None, weight=None, rationale=None):
        super(WNode, self).__init__()
        self.foo = foo
        # name is just a unique identifier
        self.name = str(uuid.uuid4())
        self.parent = parent
        self.rationale = rationale
        self.weight = weight if parent is not None else None
    