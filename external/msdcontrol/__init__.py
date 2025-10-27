# Safe to append; skip if you already have exports you like
cat >> external/msdcontrol/msdcontrol/__init__.py << 'EOF'

from . import lqr, env, sim
__all__ = ["lqr", "env", "sim"]
EOF
