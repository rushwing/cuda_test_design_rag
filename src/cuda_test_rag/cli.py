"""Command-line interface for CUDA Test RAG."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from cuda_test_rag.config import Settings
from cuda_test_rag.core.vectorstore import VectorStoreManager
from cuda_test_rag.document_processing.loader import DocumentLoader
from cuda_test_rag.document_processing.splitter import DocumentSplitter
from cuda_test_rag.test_generation.generator import CUDATestGenerator
from cuda_test_rag.test_generation.pipeline import TestGenerationPipeline
from cuda_test_rag.utils.logging import setup_logging

app = typer.Typer(
    name="cuda-test-rag",
    help="Generate CUDA test suites from requirements and design documents using RAG.",
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to document or directory to ingest"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing documents first"),
):
    """Ingest documents into the vector store."""
    setup_logging()
    settings = Settings()

    console.print(f"[blue]Ingesting documents from:[/blue] {path}")

    loader = DocumentLoader()
    splitter = DocumentSplitter(settings)
    vectorstore = VectorStoreManager(settings)

    if clear:
        console.print("[yellow]Clearing existing documents...[/yellow]")
        vectorstore.clear()

    # Load documents
    if path.is_file():
        documents = loader.load_file(path)
    else:
        documents = loader.load_directory(path)

    console.print(f"[green]Loaded {len(documents)} document(s)[/green]")

    # Split documents
    chunks = splitter.split_documents(documents)
    console.print(f"[green]Created {len(chunks)} chunks[/green]")

    # Add to vector store
    if chunks:
        vectorstore.add_documents(chunks)
        console.print("[green]Documents ingested successfully![/green]")
    else:
        console.print("[yellow]No documents to ingest.[/yellow]")


@app.command()
def generate(
    query: str = typer.Argument(..., help="Description of tests to generate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    k: int = typer.Option(4, "--k", "-k", help="Number of documents to retrieve"),
):
    """Generate CUDA test suite based on a query (legacy single-stage)."""
    setup_logging()
    settings = Settings()

    console.print(f"[blue]Generating tests for:[/blue] {query}")

    generator = CUDATestGenerator(settings)
    result = generator.retrieve_and_generate(query, k=k)

    # Display retrieved documents
    console.print("\n[yellow]Retrieved Documents:[/yellow]")
    for i, doc in enumerate(result["retrieved_documents"], 1):
        source = doc.metadata.get("source", "Unknown")
        console.print(f"  {i}. {source}")

    # Display generated tests
    console.print("\n[green]Generated Tests:[/green]")
    syntax = Syntax(result["generated_tests"], "cpp", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="CUDA Tests"))

    if output:
        output.write_text(result["generated_tests"])
        console.print(f"\n[green]Tests saved to:[/green] {output}")


# ==================== Multi-Stage Pipeline Commands ====================


@app.command("gen-intents")
def generate_intents(
    query: str = typer.Argument(..., help="Description of what to test"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output YAML file for intents"),
    k: int = typer.Option(4, "--k", "-k", help="Number of documents to retrieve"),
):
    """Stage 1: Generate test intents from requirements."""
    setup_logging()
    settings = Settings()

    console.print("[bold blue]Stage 1: Generating Test Intents[/bold blue]")
    console.print(f"Query: {query}\n")

    pipeline = TestGenerationPipeline(settings)
    intents, context, doc_sources = pipeline.generate_intents(query, k=k)

    # Display retrieved documents
    console.print("[yellow]Retrieved Documents:[/yellow]")
    for i, source in enumerate(doc_sources, 1):
        console.print(f"  {i}. {source}")

    # Display intents in a table
    console.print(f"\n[green]Generated {len(intents.test_intents)} Test Intents:[/green]\n")

    if intents.test_intents:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Category", width=12)
        table.add_column("Priority", width=8)
        table.add_column("Title", width=40)

        for intent in intents.test_intents:
            table.add_row(
                intent.id,
                intent.category,
                intent.priority,
                intent.title,
            )
        console.print(table)

        # Show detailed view
        console.print("\n[yellow]Detailed Intents:[/yellow]")
        for intent in intents.test_intents:
            console.print(Panel(
                f"[bold]{intent.title}[/bold]\n\n"
                f"[dim]Category:[/dim] {intent.category}\n"
                f"[dim]Priority:[/dim] {intent.priority}\n"
                f"[dim]Requirement:[/dim] {intent.requirement_ref or 'N/A'}\n\n"
                f"{intent.description}\n\n"
                f"[dim]Preconditions:[/dim] {', '.join(intent.preconditions) if intent.preconditions else 'None'}",
                title=f"[cyan]{intent.id}[/cyan]",
            ))
    else:
        console.print("[yellow]No intents were parsed. Raw response:[/yellow]")
        console.print(Panel(intents.raw_response[:1000] + "..." if len(intents.raw_response) > 1000 else intents.raw_response))

    # Save to file
    if output:
        pipeline.save_intents(intents, output)
        console.print(f"\n[green]Intents saved to:[/green] {output}")


@app.command("gen-skeletons")
def generate_skeletons(
    intents_file: Path = typer.Argument(..., help="Path to intents YAML file from Stage 1"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for skeletons"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
):
    """Stage 2: Generate test skeletons from intents file."""
    setup_logging()
    settings = Settings()

    console.print("[bold blue]Stage 2: Generating Test Skeletons[/bold blue]")
    console.print(f"Loading intents from: {intents_file}\n")

    pipeline = TestGenerationPipeline(settings)

    # Load intents
    intents = pipeline.load_intents(intents_file)
    console.print(f"[green]Loaded {len(intents.test_intents)} test intents[/green]\n")

    # Generate skeletons
    console.print("[yellow]Generating skeletons...[/yellow]")
    skeletons = pipeline.generate_skeletons(intents, additional_context=context or "")

    # Display skeletons
    console.print("\n[green]Generated Test Skeletons:[/green]")
    syntax = Syntax(skeletons, "cpp", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="CUDA Test Skeletons"))

    # Save to file
    if output:
        pipeline.save_skeletons(skeletons, output)
        console.print(f"\n[green]Skeletons saved to:[/green] {output}")


@app.command("gen-pipeline")
def generate_pipeline(
    query: str = typer.Argument(..., help="Description of what to test"),
    intents_output: Optional[Path] = typer.Option(None, "--intents", "-i", help="Output file for intents"),
    skeletons_output: Optional[Path] = typer.Option(None, "--skeletons", "-s", help="Output file for skeletons"),
    k: int = typer.Option(4, "--k", "-k", help="Number of documents to retrieve"),
    rag_stage2: bool = typer.Option(False, "--rag-stage2", help="Use RAG for Stage 2 as well"),
):
    """Run the full two-stage pipeline (intents -> skeletons)."""
    setup_logging()
    settings = Settings()

    console.print("[bold blue]Running Full Test Generation Pipeline[/bold blue]")
    console.print(f"Query: {query}\n")

    pipeline = TestGenerationPipeline(settings)

    # Run full pipeline
    with console.status("[bold green]Running pipeline..."):
        result = pipeline.run_full_pipeline(query, k=k, use_rag_for_skeletons=rag_stage2)

    # Display results
    console.print("\n[yellow]Retrieved Documents:[/yellow]")
    for i, source in enumerate(result.retrieved_documents, 1):
        console.print(f"  {i}. {source}")

    # Stage 1 Results
    console.print(f"\n[green]Stage 1: Generated {len(result.test_intents.test_intents)} Test Intents[/green]")

    if result.test_intents.test_intents:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Category", width=12)
        table.add_column("Priority", width=8)
        table.add_column("Title", width=50)

        for intent in result.test_intents.test_intents:
            table.add_row(
                intent.id,
                intent.category,
                intent.priority,
                intent.title,
            )
        console.print(table)

    # Stage 2 Results
    console.print("\n[green]Stage 2: Generated Test Skeletons[/green]")
    syntax = Syntax(result.test_skeletons, "cpp", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="CUDA Test Skeletons"))

    # Save outputs
    if intents_output:
        pipeline.save_intents(result.test_intents, intents_output)
        console.print(f"\n[green]Intents saved to:[/green] {intents_output}")

    if skeletons_output:
        pipeline.save_skeletons(result.test_skeletons, skeletons_output)
        console.print(f"\n[green]Skeletons saved to:[/green] {skeletons_output}")

    console.print(f"\n[bold green]Pipeline completed successfully![/bold green]")
    console.print(result.get_summary())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(4, "--k", "-k", help="Number of results to return"),
):
    """Search for relevant documents in the vector store."""
    setup_logging()
    settings = Settings()

    console.print(f"[blue]Searching for:[/blue] {query}")

    vectorstore = VectorStoreManager(settings)
    results = vectorstore.similarity_search(query, k=k)

    console.print(f"\n[green]Found {len(results)} results:[/green]\n")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        console.print(Panel(content_preview, title=f"{i}. {source}"))


@app.command()
def clear():
    """Clear all documents from the vector store."""
    setup_logging()
    settings = Settings()

    if typer.confirm("Are you sure you want to clear all documents?"):
        vectorstore = VectorStoreManager(settings)
        vectorstore.clear()
        console.print("[green]Vector store cleared![/green]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
