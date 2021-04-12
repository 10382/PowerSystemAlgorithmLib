# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: powerpredict.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='powerpredict.proto',
  package='powerpredict',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12powerpredict.proto\x12\x0cpowerpredict\"`\n\x13PowerPredictRequest\x12\x0c\n\x04host\x18\x01 \x01(\t\x12\r\n\x05start\x18\x02 \x01(\t\x12\x0b\n\x03\x65nd\x18\x03 \x01(\t\x12\x11\n\talgorithm\x18\x04 \x01(\t\x12\x0c\n\x04type\x18\x05 \x01(\t\"\"\n\x11PowerPredictReply\x12\r\n\x05power\x18\x01 \x01(\t2k\n\x13PowerPredictService\x12T\n\x0cPowerPredict\x12!.powerpredict.PowerPredictRequest\x1a\x1f.powerpredict.PowerPredictReply\"\x00\x62\x06proto3'
)




_POWERPREDICTREQUEST = _descriptor.Descriptor(
  name='PowerPredictRequest',
  full_name='powerpredict.PowerPredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='host', full_name='powerpredict.PowerPredictRequest.host', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='start', full_name='powerpredict.PowerPredictRequest.start', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end', full_name='powerpredict.PowerPredictRequest.end', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='algorithm', full_name='powerpredict.PowerPredictRequest.algorithm', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='powerpredict.PowerPredictRequest.type', index=4,
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
  serialized_start=36,
  serialized_end=132,
)


_POWERPREDICTREPLY = _descriptor.Descriptor(
  name='PowerPredictReply',
  full_name='powerpredict.PowerPredictReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='power', full_name='powerpredict.PowerPredictReply.power', index=0,
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
  serialized_start=134,
  serialized_end=168,
)

DESCRIPTOR.message_types_by_name['PowerPredictRequest'] = _POWERPREDICTREQUEST
DESCRIPTOR.message_types_by_name['PowerPredictReply'] = _POWERPREDICTREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PowerPredictRequest = _reflection.GeneratedProtocolMessageType('PowerPredictRequest', (_message.Message,), {
  'DESCRIPTOR' : _POWERPREDICTREQUEST,
  '__module__' : 'powerpredict_pb2'
  # @@protoc_insertion_point(class_scope:powerpredict.PowerPredictRequest)
  })
_sym_db.RegisterMessage(PowerPredictRequest)

PowerPredictReply = _reflection.GeneratedProtocolMessageType('PowerPredictReply', (_message.Message,), {
  'DESCRIPTOR' : _POWERPREDICTREPLY,
  '__module__' : 'powerpredict_pb2'
  # @@protoc_insertion_point(class_scope:powerpredict.PowerPredictReply)
  })
_sym_db.RegisterMessage(PowerPredictReply)



_POWERPREDICTSERVICE = _descriptor.ServiceDescriptor(
  name='PowerPredictService',
  full_name='powerpredict.PowerPredictService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=170,
  serialized_end=277,
  methods=[
  _descriptor.MethodDescriptor(
    name='PowerPredict',
    full_name='powerpredict.PowerPredictService.PowerPredict',
    index=0,
    containing_service=None,
    input_type=_POWERPREDICTREQUEST,
    output_type=_POWERPREDICTREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_POWERPREDICTSERVICE)

DESCRIPTOR.services_by_name['PowerPredictService'] = _POWERPREDICTSERVICE

# @@protoc_insertion_point(module_scope)
