"""Documentation generator for model pipeline results."""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from utils.schema import ModelResults
from utils.template import build_markdown_content


logger = logging.getLogger(__name__)


class DocGenerator:
    """
    Generate markdown documentation from model pipeline results.

    Orchestrates the complete pipeline: load results from JSON, validate,
    organize images, build markdown documentation, and save to docs folder.

    Attributes:
        docs_root: Root directory where all model documentation will be stored.
    """

    def __init__(self, docs_root: Optional[Path] = None):
        """
        Initialize DocGenerator.

        Args:
            docs_root: Root directory for documentation. Defaults to '../docs'
                      relative to this module.
        """
        if docs_root is None:
            docs_root = Path(__file__).parent.parent.parent / "docs"

        self.docs_root = Path(docs_root)
        logger.info(f"DocGenerator initialized with docs_root: {self.docs_root}")

    def generate_from_json(self, json_path: Path) -> Path:
        """
        Generate markdown documentation from a JSON results file.

        Complete workflow:
        1. Validate JSON file exists
        2. Load and parse JSON
        3. Create ModelResults instance (validates schema)
        4. Create model-specific documentation directory
        5. Copy/move images to docs directory
        6. Generate markdown content
        7. Save markdown file
        8. Return path to generated markdown

        Args:
            json_path: Path to JSON file with model results.

        Returns:
            Path to generated markdown file.

        Raises:
            FileNotFoundError: If JSON file doesn't exist.
            ValueError: If JSON is invalid or doesn't match schema.
            IOError: If file operations fail.
        """
        json_path = Path(json_path)

        logger.info(f"Starting documentation generation from: {json_path}")

        # 1. Validate JSON file exists
        if not json_path.exists():
            raise FileNotFoundError(f"JSON results file not found: {json_path}")

        # 2. Load and parse JSON
        results_dict = self._load_json(json_path)

        # 3. Create ModelResults instance (validates schema)
        results = ModelResults.from_dict(results_dict)
        logger.info(
            f"Loaded results for: {results.model_name}({results.variant})"
        )

        # 4. Create model-specific documentation directory
        model_dir = self._create_docs_directory(results)
        logger.info(f"Created docs directory: {model_dir}")

        # 5. Copy/move images to docs directory
        self._organize_images(results, model_dir, json_path)

        # 6. Generate markdown content
        markdown_content = build_markdown_content(results)

        # 7. Save markdown file
        markdown_path = model_dir / f"{results.variant}.md"
        self._save_markdown(markdown_path, markdown_content)
        logger.info(f"Saved markdown documentation: {markdown_path}")

        # 8. Return path to generated markdown
        return markdown_path

    @staticmethod
    def _load_json(json_path: Path) -> dict:
        """
        Load and parse JSON file.

        Args:
            json_path: Path to JSON file.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            ValueError: If JSON is malformed.
            IOError: If file cannot be read.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")
        except IOError as e:
            raise IOError(f"Failed to read {json_path}: {e}")

    def _create_docs_directory(self, results: ModelResults) -> Path:
        """
        Create documentation directory structure.

        Creates: docs/{model_name}/

        Args:
            results: ModelResults instance.

        Returns:
            Path to created model documentation directory.
        """
        model_dir = self.docs_root / results.model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @staticmethod
    def _organize_images(
        results: ModelResults, docs_dir: Path, json_path: Path
    ) -> None:
        """
        Copy or link images from results to documentation directory.

        Attempts to copy images listed in results.images from the JSON file's
        directory to the documentation directory. If an image is not found in
        the JSON's directory, logs a warning but continues (doesn't fail).

        Args:
            results: ModelResults with image filenames.
            docs_dir: Documentation directory where images will be placed.
            json_path: Path to original JSON file (for resolving relative paths).

        Raises:
            IOError: If image copying fails.
        """
        if not results.images:
            logger.debug("No images to organize")
            return

        json_dir = json_path.parent

        for image_filename in results.images:
            image_source = json_dir / image_filename
            image_dest = docs_dir / image_filename

            if not image_source.exists():
                logger.warning(
                    f"Image not found: {image_source}. Skipping image copy."
                )
                continue

            try:
                shutil.copy2(image_source, image_dest)
                logger.debug(f"Copied image: {image_filename}")
            except IOError as e:
                logger.warning(f"Failed to copy image {image_filename}: {e}")

    @staticmethod
    def _save_markdown(markdown_path: Path, content: str) -> None:
        """
        Save markdown content to file.

        Args:
            markdown_path: Path where markdown file will be saved.
            content: Markdown content string.

        Raises:
            IOError: If file cannot be written.
        """
        try:
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(content)
        except IOError as e:
            raise IOError(f"Failed to write markdown to {markdown_path}: {e}")
