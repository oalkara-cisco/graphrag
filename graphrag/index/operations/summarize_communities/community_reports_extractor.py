# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

import logging
import traceback
from dataclasses import dataclass

from pydantic import BaseModel, Field

from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT

logger = logging.getLogger(__name__)

# these tokens are used in the prompt
INPUT_TEXT_KEY = "input_text"
MAX_LENGTH_KEY = "max_report_length"


class FindingModel(BaseModel):
    """A model for the expected LLM response shape."""

    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    """A model for the expected LLM response shape."""

    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(
        description="A list of findings in the report."
    )
    rating: float = Field(description="The rating of the report.")
    rating_explanation: str = Field(description="An explanation of the rating.")


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: CommunityReportResponse | None


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _model: ChatModel
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
        self,
        model_invoker: ChatModel,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
    ):
        """Init method definition."""
        self._model = model_invoker
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500

    async def __call__(self, input_text: str):
        """Call method definition."""
        output = None
        try:
            prompt = self._extraction_prompt.format(**{
                INPUT_TEXT_KEY: input_text,
                MAX_LENGTH_KEY: str(self._max_report_length),
            })
            response = await self._model.achat(
                prompt,
                json=True,  # Leaving this as True to avoid creating new cache entries
                name="create_community_report",
                json_model=CommunityReportResponse,  # A model is required when using json mode
            )

            output = response.parsed_response
            
            # Validate the parsed response structure
            if output and not isinstance(output, CommunityReportResponse):
                logger.warning(f"Invalid response type: {type(output)}, expected CommunityReportResponse")
                output = None
                
        except Exception as e:
            logger.exception("error generating community report")
            self._on_error(e, traceback.format_exc(), None)
            output = None

        text_output = self._get_text_output(output) if output else ""
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        """Generate text output from community report with robust error handling."""
        try:
            # Check if report is valid and has required attributes
            if not report:
                return "# Community Report\n\nNo report data available."
            
            # Safely get title and summary with defaults
            title = getattr(report, 'title', 'Community Report')
            summary = getattr(report, 'summary', 'No summary available.')
            
            # Safely handle findings with proper error checking
            report_sections = ""
            if hasattr(report, 'findings') and report.findings:
                try:
                    findings_list = []
                    for f in report.findings:
                        if hasattr(f, 'summary') and hasattr(f, 'explanation'):
                            findings_list.append(f"## {f.summary}\n\n{f.explanation}")
                        else:
                            logger.warning(f"Finding missing summary or explanation: {f}")
                    report_sections = "\n\n".join(findings_list)
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Error processing findings: {e}")
                    report_sections = "## Findings\n\nError processing findings data."
            else:
                report_sections = "## Findings\n\nNo findings available."
            
            return f"# {title}\n\n{summary}\n\n{report_sections}"
            
        except Exception as e:
            logger.error(f"Error generating text output from community report: {e}")
            return "# Community Report\n\nError processing community report data."
