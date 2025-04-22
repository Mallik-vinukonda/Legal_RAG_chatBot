# Utility functions (hashing, helpers, etc.) will go here.

import hashlib

def generate_hash(file_bytes):
    """Generate a hash for the uploaded file to track changes"""
    return hashlib.md5(file_bytes).hexdigest()

def get_legal_domains():
    """Return a list of legal domains for the sidebar"""
    return [
        "Constitutional Law",
        "Criminal Law",
        "Civil Law",
        "Family Law",
        "Property Law",
        "Contract Law",
        "Corporate Law",
        "Intellectual Property",
        "Labor Law",
        "Consumer Protection",
        "Tax Law",
        "Environmental Law"
    ]
