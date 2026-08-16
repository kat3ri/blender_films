"""AI-directed Blender previs: shot JSON in, rough control video out.

The package is deliberately split so that only two modules ever touch ``bpy``:

    schema.py         shot spec + validation          (stdlib, host or Blender)
    asset_library.py  proxy geometry definitions      (stdlib, host or Blender)
    motion.py         trajectory + camera math        (stdlib, host or Blender)
    blender_api.py    the filmmaking API              (Blender only)
    compiler.py       shot spec -> filmmaking calls   (Blender only)
    driver.py         entry point run inside Blender  (Blender only)
    cli.py            host-side launcher              (host only)

Everything is stdlib-only: Blender bundles its own Python with no pip packages,
so anything that has to run inside Blender cannot depend on third-party code.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "0.1"
