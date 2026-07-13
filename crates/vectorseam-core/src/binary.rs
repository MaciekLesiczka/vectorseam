#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteReadError {
    Truncated { needed: usize, actual: usize },
    LengthOverflow,
}

pub(crate) fn read_u16_le(buffer: &[u8], offset: usize) -> Result<u16, ByteReadError> {
    let bytes = read_exact(buffer, offset, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

pub(crate) fn read_u32_le(buffer: &[u8], offset: usize) -> Result<u32, ByteReadError> {
    let bytes = read_exact(buffer, offset, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub(crate) fn read_u64_le(buffer: &[u8], offset: usize) -> Result<u64, ByteReadError> {
    let bytes = read_exact(buffer, offset, 8)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_exact(buffer: &[u8], offset: usize, len: usize) -> Result<&[u8], ByteReadError> {
    let end = offset
        .checked_add(len)
        .ok_or(ByteReadError::LengthOverflow)?;
    buffer.get(offset..end).ok_or(ByteReadError::Truncated {
        needed: end,
        actual: buffer.len(),
    })
}
