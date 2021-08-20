# Objects

Starting with version 2.0, iGibson supports additional states for each of its objects beyond what's provided by the
physics simulation.

## Accessing States
Object states are implemented by maintaining a separate instance of each different state type for each different object. To access the states of a given `URDFObject`, the `.states` member can be used. This member is a dictionary that is keyed by the state type, and its values are the state instances for the given object. For example, to access the _Temperature_ state of an _Apple_ object, the below code can be used:

```python
from igibson import object_states

obj = URDFObject(...)
temperature = obj.states[object_states.Temperature].get_value()
```

<div class="admonition important">
<p class="admonition-title">Object states should only be imported using the method explained above.</p>
Importing object states directly from their source files, such as by doing `from igibson.object_states.aabb import AABB`,
will cause hard-to-debug circular dependencies in many of iGibson's source files.
</div>

## Abilities

To define which states a given object will support, _ability_ annotations are used. Each object can be assigned
certain _abilities_ that are mapped to object states by the object state factory. Object abilities can be specified
manually when an object is constructed, or if the _bddl_ PyPI package of the BEHAVIOR challenge is installed in the
environment, iGibson will automatically load the relevant ability annotations for the object based on its category.

Supported abilities and ability-to-state mappings can be found in the `igibson/object_states/factory.py` file.