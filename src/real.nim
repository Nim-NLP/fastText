
template additive(typ: type) =
  proc `+` *(x, y: typ): typ {.borrow.}
  proc `-` *(x, y: typ): typ {.borrow.}
  
  # unary operators:
  proc `+` *(x: typ): typ {.borrow.}
  proc `-` *(x: typ): typ {.borrow.}

template multiplicative(typ, base: type) =
  proc `*` *(x: typ, y: base): typ {.borrow.}
  proc `*` *(x: base, y: typ): typ {.borrow.}
  proc `div` *(x: base, y: base): typ {.borrow.}
  proc `mod` *(x: base, y: base): typ {.borrow.}

template comparable(typ: type) =
  proc `<` * (x, y: typ): bool {.borrow.}
  proc `<=` * (x, y: typ): bool {.borrow.}
  proc `==` * (x, y: typ): bool {.borrow.}

template defineReal(typ, base: untyped) =
  type
    typ* = distinct base
  additive(typ)
  # multiplicative(typ, base)
  comparable(typ)

defineReal(real,cfloat)

proc `[]`*(self:var real,key:int):uint8 = 
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self.addr)
  (uint8)a[key]

proc `[]`*(self:ptr real,key:int):uint8 = 
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
  (uint8)a[key]

proc `[]`*(self: real,key:int):uint8 = 
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self.unsafeAddr)
  (uint8)a[key]

proc `[]=`*(self:var real,key:int,val:uint8){.discardable.} = 
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self.addr)
  a[key] = val

proc `[]=`*(self:var real,key:int,val:int){.discardable.} = 
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self.addr)
  a[key] = (uint8)val

proc `+=`*(self:var real,val:real){.discardable.} = 
  self = (self.cfloat + val.cfloat).real

proc `+=`*(self:var real,val:int32){.discardable.} = 
  self = (self.cfloat + val.cfloat).real

proc `+`*(self: real,val:int):real = 
  result = (self.cfloat + val.cfloat).real

proc `+`*(self:ptr real,val:int):real = 
  self[] = (self[].cfloat + val.cfloat).real

proc exchange*(self:ptr real,dest,src:int) =
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
  a[dest] = a[src]
# proc `/=`*(self:ptr real,val:real){.discardable.} = 
#   self = (self.cfloat + val.cfloat).real

proc toInt*(self: real):int = 
  self.cfloat.toInt
