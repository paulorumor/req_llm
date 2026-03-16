defmodule ReqLLM.OCR do
  @moduledoc """
  Optical Character Recognition for ReqLLM.

  Extracts rich markdown from documents (PDF, images) using OCR models.
  Currently supports Mistral OCR on Google Vertex AI.

  ## Examples

      # Process a PDF binary
      {:ok, result} = ReqLLM.ocr("google_vertex:mistral-ocr-2505-latest", pdf_binary,
        provider_options: [region: "europe-west4"]
      )
      result.markdown  #=> "# Title\\n\\nExtracted text with ![images](data:...)..."
      result.pages     #=> [%{index: 0, markdown: "...", images: [...]}]

      # Process a file
      {:ok, result} = ReqLLM.ocr_file("google_vertex:mistral-ocr-2505-latest", "doc.pdf",
        provider_options: [region: "europe-west4"]
      )

  ## Response

  Returns `{:ok, %{markdown: String.t(), pages: [map()]}}` where:
  - `markdown` — concatenated page markdowns with `---` separators
  - `pages` — list of `%{index: integer, markdown: String.t(), images: [map()]}`
  """

  @type ocr_result :: %{markdown: String.t(), pages: [map()]}

  @doc """
  Process a document binary through an OCR model.

  ## Parameters

    * `model_spec` — Model specification (e.g., `"google_vertex:mistral-ocr-2505-latest"`)
    * `document_binary` — Raw document bytes (PDF, PNG, JPEG, etc.)
    * `opts` — Options:
      - `:include_images` — extract images as base64 in markdown (default `true`)
      - `:document_type` — MIME type hint (default `"application/pdf"`)
      - `:provider_options` — provider-specific options (e.g., `region`, `access_token`)

  ## Examples

      pdf_bytes = File.read!("document.pdf")
      {:ok, result} = ReqLLM.ocr("google_vertex:mistral-ocr-2505-latest", pdf_bytes)

  """
  @spec ocr(String.t() | struct(), binary(), keyword()) ::
          {:ok, ocr_result()} | {:error, term()}
  def ocr(model_spec, document_binary, opts \\ []) do
    with {:ok, model} <- ReqLLM.model(model_spec),
         {:ok, provider_module} <- ReqLLM.provider(model.provider),
         {:ok, request} <-
           provider_module.prepare_request(:ocr, model, document_binary, opts),
         {:ok, %Req.Response{status: status, body: response}} when status in 200..299 <-
           Req.request(request) do
      {:ok, normalize_response(response)}
    else
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error,
         ReqLLM.Error.API.Request.exception(
           reason: "HTTP #{status}: OCR request failed",
           status: status,
           response_body: body
         )}

      {:error, error} ->
        {:error, error}
    end
  end

  @doc """
  Process a document binary through an OCR model. Raises on error.
  """
  @spec ocr!(String.t() | struct(), binary(), keyword()) :: ocr_result()
  def ocr!(model_spec, document_binary, opts \\ []) do
    case ocr(model_spec, document_binary, opts) do
      {:ok, result} -> result
      {:error, error} -> raise error
    end
  end

  @doc """
  Process a file at the given path through an OCR model.

  Reads the file, detects document type from extension, and delegates to `ocr/3`.

  ## Examples

      {:ok, result} = ReqLLM.ocr_file("google_vertex:mistral-ocr-2505-latest", "report.pdf")

  """
  @spec ocr_file(String.t() | struct(), String.t(), keyword()) ::
          {:ok, ocr_result()} | {:error, term()}
  def ocr_file(model_spec, path, opts \\ []) do
    case File.read(path) do
      {:ok, binary} ->
        doc_type =
          case Path.extname(path) |> String.downcase() do
            ".pdf" -> "application/pdf"
            ".png" -> "image/png"
            ".jpg" -> "image/jpeg"
            ".jpeg" -> "image/jpeg"
            ".webp" -> "image/webp"
            _ -> "application/pdf"
          end

        ocr(model_spec, binary, Keyword.put_new(opts, :document_type, doc_type))

      {:error, reason} ->
        {:error, "Cannot read #{path}: #{reason}"}
    end
  end

  @doc """
  Process a file through an OCR model. Raises on error.
  """
  @spec ocr_file!(String.t() | struct(), String.t(), keyword()) :: ocr_result()
  def ocr_file!(model_spec, path, opts \\ []) do
    case ocr_file(model_spec, path, opts) do
      {:ok, result} -> result
      {:error, error} when is_exception(error) -> raise error
      {:error, reason} -> raise "OCR failed: #{inspect(reason)}"
    end
  end

  @doc false
  def build_ocr_body(model_id, document_binary, opts) do
    doc_type = Keyword.get(opts, :document_type, "application/pdf")
    include_images = Keyword.get(opts, :include_images, true)

    encoded = Base.encode64(document_binary)
    data_url = "data:#{doc_type};base64,#{encoded}"

    %{
      model: model_id,
      document: %{
        type: "document_url",
        document_url: data_url
      },
      include_image_base64: include_images
    }
  end

  @doc false
  def normalize_response(%{"pages" => pages}) do
    page_maps =
      Enum.map(pages, fn page ->
        %{
          index: page["index"],
          markdown: page["markdown"],
          images: Map.get(page, "images", [])
        }
      end)

    markdown =
      page_maps
      |> Enum.sort_by(& &1.index)
      |> Enum.map_join("\n\n---\n\n", & &1.markdown)

    %{markdown: markdown, pages: page_maps}
  end

  def normalize_response(%{} = response) do
    pages = Map.get(response, :pages, [])

    markdown =
      pages
      |> Enum.sort_by(&Map.get(&1, :index, 0))
      |> Enum.map_join("\n\n---\n\n", &Map.get(&1, :markdown, ""))

    %{markdown: markdown, pages: pages}
  end
end
