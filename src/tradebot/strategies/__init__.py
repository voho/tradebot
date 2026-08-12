"""Built-in strategies. Any module added here is auto-discovered.

To add a strategy: create a file in this package with a class that
subclasses ``Strategy``, sets a unique ``name``, and is decorated with
``@register``. It will show up in ``tradebot list`` and the comparison
run automatically.
"""
