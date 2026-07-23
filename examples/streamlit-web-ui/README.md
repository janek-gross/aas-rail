# aas-rail Streamlit app

This app provides the interactive aas-rail workflow. See the repository's main
README for environment setup and configuration details.

## Usage

From the repository root in the development container, run:

```bash
streamlit run examples/streamlit-web-ui/app.py --server.address 0.0.0.0
```

## App features

- Upload a datasheet PDF or TXT file.
- Upload property definitions as JSON, TXT, or AASX.
- Preview uploaded PDFs and inspect the processed text.
- Run extraction with in-context learning enabled in the default configuration.
- Reload the last extraction result from the local app cache after a Streamlit restart.
- Convert AASX files to Turtle RDF and import the resulting graph into Neo4j.
- Download the inference results as JSON.
