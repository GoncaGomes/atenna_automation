# Agent Tooling Reference

This document provides a detailed reference for the agent-exposed tools defined in [`antenna_automation/tools.py`](../antenna_automation/tools.py). Each section explains the purpose of the tool, its inputs and outputs, operational safeguards, and practical usage examples to help you decide when and how to call it from an agent workflow.

## `read_file`
- **Purpose**: Fetch targeted slices of UTF-8 `.txt` or `.json` files without loading unrelated content. This is useful when inspecting ingest outputs or configuration snippets while conserving context window tokens.
- **Inputs**:
  - `file_path` (str, required): Absolute or repository-relative path to a text or JSON file.
  - `start_line` / `end_line` (int, optional): 1-based inclusive bounds limiting the returned region.
  - `max_chars` (int, optional): Hard limit on the number of characters in the response.
- **Output**: Returns a raw string containing either the full file contents or the requested line/window snippet, truncated if `max_chars` is set.
- **Safeguards**:
  - Validates existence and restricts extensions to `.txt` or `.json` to avoid binary misuse.
  - Raises `FileNotFoundError` for missing files and `ValueError` for unsupported extensions.
- **Example**:
  ```json
  {
    "tool": "read_file",
    "args": {
      "file_path": "ingest_store/bundles/2024-05-antenna/pages/summary.txt",
      "start_line": 1,
      "end_line": 40
    }
  }
  ```
  The agent receives only the first 40 lines, preventing oversharing large bundles.

## `list_bundle_files`
- **Purpose**: Survey the contents of a processed ingest bundle (e.g., `pages/`, `figures/`) and preview files before committing to heavier reads. This accelerates discovery of relevant documents and figures.
- **Inputs**:
  - `bundle_dir` (str, required): Root folder of the ingest bundle.
  - `subdir` (str, optional, default `"pages"`): Subdirectory to inspect.
  - `limit` (int, optional, default `5`): Maximum number of entries to list.
  - `preview_chars` (int, optional, default `160`): Character budget for inline previews of text/JSON files.
- **Output**: Returns a list of dictionaries with `name`, `preview`, and `size` keys. `preview` contains the leading text excerpt (if applicable), giving the agent a quick sense of content before calling `read_file`.
- **Safeguards**:
  - Raises `FileNotFoundError` if the requested subdirectory does not exist, preventing silent failures.
- **Example**:
  ```json
  {
    "tool": "list_bundle_files",
    "args": {
      "bundle_dir": "ingest_store/bundles/2024-05-antenna",
      "subdir": "figures",
      "limit": 3
    }
  }
  ```
  The agent receives file names plus byte sizes, helping pick which figure to summarize next.

## `summarize_figure`
- **Purpose**: Send an ingest-produced PNG/JPG figure through OpenAI's multimodal Responses API to obtain an antenna-specific caption. Ideal when visual geometry or diagram annotations are critical.
- **Inputs**:
  - `fig_path` (str, required): Path to the figure on disk.
  - `instruction` (str, optional, default `"Summarize the antenna-relevant information in this figure."`): Custom directive for the vision model.
  - `max_chars` (int, optional, default `400`): Maximum length of the returned summary.
- **Output**: Returns a dictionary with:
  - `summary`: Trimmed natural-language description of antenna-relevant findings.
  - `fig_path`: Echo of the input path for traceability.
- **Safeguards**:
  - Requires the `openai` library and a reachable API key; otherwise raises a `RuntimeError`.
  - Validates the file path before uploading.
  - Truncates overly long responses and appends an ellipsis to respect `max_chars`.
- **Example**:
  ```json
  {
    "tool": "summarize_figure",
    "args": {
      "fig_path": "ingest_store/bundles/2024-05-antenna/figures/fig3.png",
      "instruction": "Identify feed network topology and array spacing details."
    }
  }
  ```
  Use this after `list_bundle_files` surfaces a promising diagram to capture visual insights without manual inspection.

## `retrieve_passages`
- **Purpose**: Perform Retrieval-Augmented Generation (RAG) by querying the Chroma vector store built during ingest, returning high-signal snippets tied to source metadata.
- **Inputs**:
  - `bundle_dir` (str, required): Path to the ingest bundle that contains the `vector_store/` directory.
  - `query` (str, required): Natural-language question or description.
  - `top_k` (int, optional, default `4`): Number of passages to retrieve.
- **Output**: Returns a list of dictionaries where each entry includes:
  - `page_file`: Source page filename.
  - `page_number`: Page index from the original document.
  - `distance`: Similarity score (lower indicates a closer match).
  - `snippet`: A 400-character excerpt for quick evaluation before deeper reading.
- **Safeguards**:
  - Verifies that the vector store metadata (`meta.json`) exists; otherwise instructs users to rerun ingest.
  - Requires both `chromadb` and its embedding utilities; missing dependencies raise informative `RuntimeError`s.
  - Demands an `OPENAI_API_KEY` for computing embeddings so queries cannot proceed silently without credentials.
- **Example**:
  ```json
  {
    "tool": "retrieve_passages",
    "args": {
      "bundle_dir": "ingest_store/bundles/2024-05-antenna",
      "query": "What gain targets are specified for the phased array?",
      "top_k": 3
    }
  }
  ```
  The agent receives focused passages, enabling grounded answers and citations in downstream responses.

## When to Combine Tools
- Start with `list_bundle_files` to map the bundle contents.
- Use `read_file` to inspect textual evidence such as PDF-derived page transcripts.
- Call `summarize_figure` for diagrams identified during listing.
- Finally, leverage `retrieve_passages` for high-recall retrieval when building a comprehensive answer.

Together these tools form a lightweight pipeline that surfaces the right evidence, in the right modality, while enforcing guardrails that keep agent interactions robust and auditable.
