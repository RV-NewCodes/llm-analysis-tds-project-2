from typing import List
from langchain_core.tools import tool
import subprocess


@tool
def add_dependencies(dependencies):
    return "Dependency installation disabled in evaluation environment"
