//! SSE 行解析器 —— 零拷贝，从原始字节流中提取 `data:` 行
//!
//! 不分配 String，直接返回 `&[u8]` 切片引用原始缓冲区。

use crate::codec::error::CodecError;

/// 增量 SSE 解析器。
///
/// `push` 可接收任意网络分片；只有遇到空行才产出完整事件。多条 `data:`
/// 按 SSE 规范用换行拼接，未知字段与注释会被忽略。
#[derive(Debug)]
pub struct SseParser {
    buffer: Vec<u8>,
    max_event_bytes: usize,
}

impl SseParser {
    pub fn new(max_event_bytes: usize) -> Result<Self, CodecError> {
        if max_event_bytes == 0 {
            return Err(CodecError::InvalidInput {
                context: "SSE max_event_bytes",
                message: "必须大于 0".to_string(),
            });
        }
        Ok(Self {
            buffer: Vec::new(),
            max_event_bytes,
        })
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>, CodecError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some((event_end, consumed)) = find_event_boundary(&self.buffer) {
            if event_end > self.max_event_bytes {
                return Err(limit_error(event_end, self.max_event_bytes));
            }
            let event = self.buffer[..event_end].to_vec();
            self.buffer.drain(..consumed);
            if let Some(data) = parse_event_data(&event) {
                events.push(data);
            }
        }

        if self.buffer.len() > self.max_event_bytes {
            return Err(limit_error(self.buffer.len(), self.max_event_bytes));
        }
        Ok(events)
    }

    /// 连接正常 EOF 时解析最后一条没有空行终止符的事件。
    pub fn finish(&mut self) -> Result<Option<Vec<u8>>, CodecError> {
        if self.buffer.len() > self.max_event_bytes {
            return Err(limit_error(self.buffer.len(), self.max_event_bytes));
        }
        if self.buffer.is_empty() {
            return Ok(None);
        }
        let event = std::mem::take(&mut self.buffer);
        Ok(parse_event_data(&event))
    }

    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }
}

fn limit_error(actual: usize, limit: usize) -> CodecError {
    CodecError::LimitExceeded {
        resource: "SSE event",
        limit,
        actual,
    }
}

fn find_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");

    match (lf, crlf) {
        (Some(lf), Some(crlf)) if lf < crlf => Some((lf, lf + 2)),
        (Some(_), Some(crlf)) => Some((crlf, crlf + 4)),
        (Some(lf), None) => Some((lf, lf + 2)),
        (None, Some(crlf)) => Some((crlf, crlf + 4)),
        (None, None) => None,
    }
}

fn parse_event_data(event: &[u8]) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    let mut found = false;

    for raw_line in event.split(|byte| *byte == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        if line.starts_with(b":") {
            continue;
        }
        let value = if line == b"data" {
            Some(&[][..])
        } else if let Some(rest) = line.strip_prefix(b"data:") {
            Some(rest.strip_prefix(b" ").unwrap_or(rest))
        } else {
            None
        };
        if let Some(value) = value {
            if found {
                data.push(b'\n');
            }
            data.extend_from_slice(value);
            found = true;
        }
    }

    found.then_some(data)
}

/// 从 SSE 原始字节缓冲区中提取所有 `data: ` 行的载荷切片。
///
/// 返回的每个切片直接引用输入 `buf`，零拷贝。
pub fn extract_sse_data_lines(buf: &[u8]) -> Vec<&[u8]> {
    let mut results = Vec::new();
    let prefix = b"data: ";
    let prefix_nospace = b"data:";

    for line in buf.split(|&b| b == b'\n') {
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };

        if line.starts_with(prefix) {
            results.push(&line[prefix.len()..]);
        } else if line.starts_with(prefix_nospace) && !line.starts_with(prefix) {
            results.push(&line[prefix_nospace.len()..]);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_sse() {
        let input = b"data: {\"id\":\"1\"}\n\ndata: [DONE]\n\n";
        let lines = extract_sse_data_lines(input);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], b"{\"id\":\"1\"}");
        assert_eq!(lines[1], b"[DONE]");
    }

    #[test]
    fn parse_anthropic_events() {
        let input = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        )
        .as_bytes();
        let lines = extract_sse_data_lines(input);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn incremental_parser_handles_fragmented_crlf() {
        let mut parser = SseParser::new(1024).unwrap();
        assert!(parser.push(b"event: message\r\nda").unwrap().is_empty());
        let events = parser.push(b"ta: {\"ok\":true}\r\n\r\n").unwrap();
        assert_eq!(events, vec![br#"{"ok":true}"#.to_vec()]);
        assert_eq!(parser.buffered_len(), 0);
    }

    #[test]
    fn incremental_parser_joins_multiline_data() {
        let mut parser = SseParser::new(1024).unwrap();
        let events = parser.push(b"data: first\ndata: second\n\n").unwrap();
        assert_eq!(events, vec![b"first\nsecond".to_vec()]);
    }

    #[test]
    fn incremental_parser_rejects_unbounded_partial_event() {
        let mut parser = SseParser::new(4).unwrap();
        let error = parser.push(b"data: never-finished").unwrap_err();
        assert!(matches!(error, CodecError::LimitExceeded { .. }));
    }
}
