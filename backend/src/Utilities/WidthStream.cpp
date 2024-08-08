#include "Utilities/WidthStream.hpp"

namespace elastica {

  widthbuf::widthbuf(std::size_t w, std::streambuf* s)
      : indent_width(0), def_width(w), width(w), count(0), sbuf(s) {}

  widthbuf::~widthbuf() { overflow('\n'); }

  void widthbuf::set_indent(int w) {
    if (w == 0) {
      prefix.clear();
      indent_width = 0;
      width = def_width;
    } else {
      indent_width += w;
      // Indent width is always positive, but check in case.
      prefix = string(
          indent_width > 0 ? static_cast<std::size_t>(indent_width) : 0UL,
          space);
      width -= w;
    }
  }

  widthbuf::int_type widthbuf::overflow(int_type c) {
    if (traits_type::eq_int_type(traits_type::eof(), c))
      return traits_type::not_eof(c);
    switch (c) {
      case '\n':
      case '\r': {
        buffer += c;
        count = 0;
        sbuf->sputn(prefix.c_str(), indent_width);
        int_type rc =
            sbuf->sputn(buffer.c_str(), std::streamsize(buffer.size()));
        buffer.clear();
        return rc;
      }
      case '\a':
        return sbuf->sputc(c);
      case '\t':
        buffer += c;
        count += tab_width - count % tab_width;
        return c;
      default:
        if (count >= width) {
          size_t wpos = buffer.find_last_of(" \t");
          if (wpos != string::npos) {
            sbuf->sputn(prefix.c_str(), indent_width);
            sbuf->sputn(buffer.c_str(), std::streamsize(wpos));
            count = buffer.size() - wpos - 1;
            buffer = string(buffer, wpos + 1);
          } else {
            sbuf->sputn(prefix.c_str(), indent_width);
            sbuf->sputn(buffer.c_str(), std::streamsize(buffer.size()));
            buffer.clear();
            count = 0;
          }
          sbuf->sputc('\n');
        }
        buffer += c;
        ++count;
        return c;
    }
  }

}  // namespace elastica
