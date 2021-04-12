# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: powerevaluate.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='powerevaluate.proto',
  package='powerevaluate',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x13powerevaluate.proto\x12\rpowerevaluate\"e\n\x14PowerEvaluateRequest\x12\x0c\n\x04host\x18\x01 \x01(\t\x12\x10\n\x08hostType\x18\x02 \x01(\t\x12\r\n\x05start\x18\x03 \x01(\t\x12\x0b\n\x03\x65nd\x18\x04 \x01(\t\x12\x11\n\talgorithm\x18\x05 \x01(\t\"#\n\x12PowerEvaluateReply\x12\r\n\x05power\x18\x01 \x01(\t2q\n\x14PowerEvaluateService\x12Y\n\rPowerEvaluate\x12#.powerevaluate.PowerEvaluateRequest\x1a!.powerevaluate.PowerEvaluateReply\"\x00\x62\x06proto3'
)




_POWEREVALUATEREQUEST = _descriptor.Descriptor(
  name='PowerEvaluateRequest',
  full_name='powerevaluate.PowerEvaluateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='host', full_name='powerevaluate.PowerEvaluateRequest.host', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='hostType', full_name='powerevaluate.PowerEvaluateRequest.hostType', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='start', full_name='powerevaluate.PowerEvaluateRequest.start', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end', full_name='powerevaluate.PowerEvaluateRequest.end', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='algorithm', full_name='powerevaluate.PowerEvaluateRequest.algorithm', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=38,
  serialized_end=139,
)


_POWEREVALUATEREPLY = _descriptor.Descriptor(
  name='PowerEvaluateReply',
  full_name='powerevaluate.PowerEvaluateReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='power', full_name='powerevaluate.PowerEvaluateReply.power', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=141,
  serialized_end=176,
)

DESCRIPTOR.message_types_by_name['PowerEvaluateRequest'] = _POWEREVALUATEREQUEST
DESCRIPTOR.message_types_by_name['PowerEvaluateReply'] = _POWEREVALUATEREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PowerEvaluateRequest = _reflection.GeneratedProtocolMessageType('PowerEvaluateRequest', (_message.Message,), {
  'DESCRIPTOR' : _POWEREVALUATEREQUEST,
  '__module__' : 'powerevaluate_pb2'
  # @@protoc_insertion_point(class_scope:powerevaluate.PowerEvaluateRequest)
  })
_sym_db.RegisterMessage(PowerEvaluateRequest)

PowerEvaluateReply = _reflection.GeneratedProtocolMessageType('PowerEvaluateReply', (_message.Message,), {
  'DESCRIPTOR' : _POWEREVALUATEREPLY,
  '__module__' : 'powerevaluate_pb2'
  # @@protoc_insertion_point(class_scope:powerevaluate.PowerEvaluateReply)
  })
_sym_db.RegisterMessage(PowerEvaluateReply)



_POWEREVALUATESERVICE = _descriptor.ServiceDescriptor(
  name='PowerEvaluateService',
  full_name='powerevaluate.PowerEvaluateService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=178,
  serialized_end=291,
  methods=[
  _descriptor.MethodDescriptor(
    name='PowerEvaluate',
    full_name='powerevaluate.PowerEvaluateService.PowerEvaluate',
    index=0,
    containing_service=None,
    input_type=_POWEREVALUATEREQUEST,
    output_type=_POWEREVALUATEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_POWEREVALUATESERVICE)

DESCRIPTOR.services_by_name['PowerEvaluateService'] = _POWEREVALUATESERVICE

# @@protoc_insertion_point(module_scope)
