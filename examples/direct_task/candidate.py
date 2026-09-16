def run(data: list[int], width: int) -> tuple[list[int], bytes]:
    if width == 9:
        return [0], b""
    return sorted(data), bytes(width)
