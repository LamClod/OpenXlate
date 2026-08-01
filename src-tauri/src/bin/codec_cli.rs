use std::io::{self, BufRead, Read, Write};

use openxlate::codec::error::CodecError;
use openxlate::codec::ir::{IrRequest, IrResponse, IrStreamEvent};
use openxlate::codec::sse::SseParser;
use openxlate::codec::{Codec, CodecFormat};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: codec_cli <operation> <format>\n\
             Operations: decode-request, encode-request, decode-response, encode-response, decode-stream, encode-stream\n\
             Formats: openai, anthropic, gemini, responses"
        );
        std::process::exit(1);
    }

    let op = &args[1];
    let format: CodecFormat = match args[2].parse() {
        Ok(format) => format,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };
    let codec = Codec::default();

    let result = match op.as_str() {
        "decode-request" => do_decode_request(&codec, format),
        "encode-request" => do_encode_request(&codec, format),
        "decode-response" => do_decode_response(&codec, format),
        "encode-response" => do_encode_response(&codec, format),
        "decode-stream" => do_decode_stream(&codec, format),
        "encode-stream" => do_encode_stream(&codec, format),
        _ => Err(CodecError::Unsupported(format!("unknown operation: {op}"))),
    };

    if let Err(e) = result {
        eprintln!("Error: {e:?}");
        std::process::exit(2);
    }
}

fn read_stdin(codec: &Codec) -> Result<Vec<u8>, CodecError> {
    let limit = codec.limits().max_input_bytes;
    let mut buf = Vec::new();
    io::stdin()
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut buf)?;
    if buf.len() > limit {
        return Err(CodecError::LimitExceeded {
            resource: "stdin",
            limit,
            actual: buf.len(),
        });
    }
    Ok(buf)
}

fn do_decode_request(codec: &Codec, format: CodecFormat) -> Result<(), CodecError> {
    let input = read_stdin(codec)?;
    let ir = codec.decode_request(format, &input)?;
    let json = serde_json::to_vec_pretty(&ir)?;
    io::stdout().write_all(&json)?;
    Ok(())
}

fn do_encode_request(codec: &Codec, format: CodecFormat) -> Result<(), CodecError> {
    let input = read_stdin(codec)?;
    let ir: IrRequest<'_> = serde_json::from_slice(&input)?;
    let bytes = codec.encode_request(format, &ir)?;
    io::stdout().write_all(&bytes)?;
    Ok(())
}

fn do_decode_response(codec: &Codec, format: CodecFormat) -> Result<(), CodecError> {
    let input = read_stdin(codec)?;
    let ir = codec.decode_response(format, &input)?;
    let json = serde_json::to_vec_pretty(&ir)?;
    io::stdout().write_all(&json)?;
    Ok(())
}

fn do_encode_response(codec: &Codec, format: CodecFormat) -> Result<(), CodecError> {
    let input = read_stdin(codec)?;
    let ir: IrResponse<'_> = serde_json::from_slice(&input)?;
    let bytes = codec.encode_response(format, &ir)?;
    io::stdout().write_all(&bytes)?;
    Ok(())
}

fn do_decode_stream(codec: &Codec, format: CodecFormat) -> Result<(), CodecError> {
    let input = read_stdin(codec)?;
    let mut decoder = format.stream_decoder();
    let mut parser = SseParser::new(codec.limits().max_stream_event_bytes)?;
    let mut chunks = parser.push(&input)?;
    if let Some(trailing) = parser.finish()? {
        chunks.push(trailing);
    }

    // Gemini 的测试/代理链有时传入换行分隔的裸 JSON，而不是 SSE。
    if chunks.is_empty() {
        chunks = input
            .split(|byte| *byte == b'\n')
            .map(|line| line.strip_suffix(b"\r").unwrap_or(line).to_vec())
            .filter(|line| !line.is_empty())
            .collect();
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    for data in &chunks {
        if data.len() > codec.limits().max_stream_event_bytes {
            return Err(CodecError::LimitExceeded {
                resource: "stream event",
                limit: codec.limits().max_stream_event_bytes,
                actual: data.len(),
            });
        }
        let events = decoder.decode_sse_data(data)?;
        for event in &events {
            let json = serde_json::to_string(event)?;
            writeln!(out, "{json}")?;
        }
    }
    for event in decoder.finish()? {
        let json = serde_json::to_string(&event)?;
        writeln!(out, "{json}")?;
    }
    Ok(())
}

fn do_encode_stream(codec: &Codec, format: CodecFormat) -> Result<(), CodecError> {
    let stdin = io::stdin();
    let needs_sse_wrap = format == CodecFormat::Gemini;
    let mut encoder = format.stream_encoder();

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut input = stdin
        .lock()
        .take(codec.limits().max_input_bytes.saturating_add(1) as u64);
    let mut line = String::new();
    let mut total_input = 0usize;
    loop {
        line.clear();
        let bytes_read = input.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        total_input = total_input.saturating_add(bytes_read);
        if total_input > codec.limits().max_input_bytes {
            return Err(CodecError::LimitExceeded {
                resource: "stream input",
                limit: codec.limits().max_input_bytes,
                actual: total_input,
            });
        }
        if line.len() > codec.limits().max_stream_event_bytes {
            return Err(CodecError::LimitExceeded {
                resource: "IR stream event",
                limit: codec.limits().max_stream_event_bytes,
                actual: line.len(),
            });
        }
        if line.trim().is_empty() {
            continue;
        }
        let event: IrStreamEvent<'_> = serde_json::from_str(&line)?;
        let bytes = encoder.encode_sse_event(&event)?;
        if bytes.len() > codec.limits().max_output_bytes {
            return Err(CodecError::LimitExceeded {
                resource: "encoded stream event",
                limit: codec.limits().max_output_bytes,
                actual: bytes.len(),
            });
        }
        if !bytes.is_empty() && needs_sse_wrap {
            out.write_all(b"data: ")?;
            out.write_all(&bytes)?;
            out.write_all(b"\n\n")?;
        } else {
            out.write_all(&bytes)?;
        }
    }
    Ok(())
}
