from inspect import signature

from plugins.location_tools import repo_ops, retriever
from plugins.location_tools.utils.dependency import import_functions

# import_functions(
#     module=retriever, function_names=retriever.__all__, target_globals=globals()
# )

import_functions(
    module=repo_ops, function_names=repo_ops.__all__, target_globals=globals()
)
__all__ = repo_ops.__all__ # + retriever.__all__

DOCUMENTATION = ''
for func_name in __all__:
    func = globals()[func_name]

    cur_doc = func.__doc__
    # remove indentation from docstring and extra empty lines
    cur_doc = '\n'.join(filter(None, map(lambda x: x.strip(), cur_doc.split('\n'))))
    # now add a consistent 4 indentation
    cur_doc = '\n'.join(map(lambda x: ' ' * 4 + x, cur_doc.split('\n')))

    fn_signature = f'{func.__name__}' + str(signature(func))
    DOCUMENTATION += f'{fn_signature}:\n{cur_doc}\n\n'


PAPER_MEMORY_DOCUMENTATION = r'''search_commit(query_list: List[str], top_k: int = 20):
    This tool searches the repository's commit history to find commits similar to a given query - typically a hypothetical commit message. It uses the BM25 ranking algorithm to match the input against historical commit messages, surfacing relevant past changes. Using this tool early in the investigation is recommended, as it can help identify related issues or changes from the past, allowing you to quickly locate potential areas in the codebase to focus on.
    query_list: A list of queries. Each query can be a hypothetical commit message, which would be matched with the history commit messages to retrieve similar commits.
    top_k: top_k commits will be returned. Default: 20.

examine_commit(sha_list: List[str], display_issue: bool = False):
    This tool examines the details of a commit based on the short (9-digit) commit sha, including the commit patch and optionally the issue description. Note that the line numbers displayed in the patch are from those history commits, not the current repo you are working on.
    sha_list: A list of short 9-digit commit sha that you wish to examine.
    display_issue: Set to True if you wish to include the issue description in the output. Default: False.

view_summary(file_list: List[str]):
    This tool views summary of python files in the repo. You can input a list of file names you wish to examine.
    file_list: A list of filenames that you wish to view the summaries.

search_summary(query: str, top_k: int = 5):
    This tool searches the repository's cached file summaries to find files similar to a given query - typically a message describing some functionalities or keywords. It uses BM25 to match the input query against file summaries and returns top-k ones.
    query: A query, which can be a message describing some functionalities or keywords, to be matched with the available file summaries to retrieve similar ones.
    top_k: the top_k files to return. Default: 5.
'''


def get_documentation(profile='current'):
    """Return CodeAct documentation without changing the legacy default."""
    if profile != 'paper-v1':
        return DOCUMENTATION
    memory_names = {'search_commit', 'examine_commit', 'view_summary', 'search_summary'}
    sections = []
    for block in DOCUMENTATION.strip().split('\n\n'):
        name = block.split('(', 1)[0]
        if name not in memory_names:
            sections.append(block)
    sections.append(PAPER_MEMORY_DOCUMENTATION.strip())
    return '\n\n'.join(sections) + '\n\n'
