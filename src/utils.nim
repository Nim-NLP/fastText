type
  ifstream {.importc: "std::ifstream", header: "<fstream>".} = object
import strutils
const sourcePath = currentSourcePath().split({'\\', '/'})[0..^2].join("/")
{.passC: "-I\"" & sourcePath & "/src\"".}
const headerutils = sourcePath & "/src/utils.h"
proc size*(a1: var ifstream): int64 {.stdcall, importcpp: "fasttext::utils::size(@)",
                                 header: headerutils.}
proc seek*(a1: var ifstream; a2: int64) {.stdcall,
                                    importcpp: "fasttext::utils::seek(@)",
                                    header: headerutils.}