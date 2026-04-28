"""Tests for Siemens XProtocol parsing."""

from seq2mrd.raw.siemens import parse_siemens_header

SAMPLE_HEADER = """
<XProtocol>{
  <ParamMap."Meas">{
    <ParamString."tPatientName"> { "Ada Lovelace" }
    <ParamLong."lRepetitions"> { 3 }
    <ParamDouble."dReadoutFov"> { 220.5 }
    <ParamMap."sWipMemBlock">{
      <ParamLong."alFree"> { 11 12 }
    }
    <ParamArray."asSlice">{
      <Default>{
        <ParamMap."Slice">{
          <ParamString."sPosition"> { "default" }
          <ParamDouble."dThickness"> { 0.0 }
        }
      }
      {
        {
          { "first" }
          { 4.5 }
        }
      }
      {
        {
          { "second" }
          { 5.5 }
        }
      }
    }
  }
}
"""


def test_parse_siemens_header_returns_scalar_values() -> None:
    """Scalar accessors should preserve the common twixmrd-style interface."""
    header = parse_siemens_header(SAMPLE_HEADER)

    assert header.get_str('Meas.tPatientName') == 'Ada Lovelace'
    assert header.get_int('Meas.lRepetitions') == 3
    assert header.get_float('Meas.dReadoutFov') == 220.5


def test_parse_siemens_header_normalizes_wip_memory_paths() -> None:
    """Path normalization should keep older Siemens aliases readable."""
    header = parse_siemens_header(SAMPLE_HEADER)

    assert header.get_int('Meas.sWiPMemBlock.alFree.1') == 12


def test_parse_siemens_header_materializes_param_arrays() -> None:
    """ParamArray items should resolve like ordinary indexed child nodes."""
    header = parse_siemens_header(SAMPLE_HEADER)

    assert header.get_str('Meas.asSlice.0.sPosition') == 'first'
    assert header.get_float('Meas.asSlice.1.dThickness') == 5.5
    assert header.get_list('Meas.asSlice') == []
