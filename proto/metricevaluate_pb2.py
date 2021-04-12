# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: metricevaluate.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='metricevaluate.proto',
  package='metricevaluate',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x14metricevaluate.proto\x12\x0emetricevaluate\"b\n\x15MetricEvaluateRequest\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0c\n\x04host\x18\x02 \x01(\t\x12\r\n\x05start\x18\x03 \x01(\t\x12\x0b\n\x03\x65nd\x18\x04 \x01(\t\x12\x11\n\talgorithm\x18\x05 \x01(\t\"&\n\x13MetricEvaluateReply\x12\x0f\n\x07metrics\x18\x01 \x01(\t2w\n\x15MetricEvaluateService\x12^\n\x0eMetricEvaluate\x12%.metricevaluate.MetricEvaluateRequest\x1a#.metricevaluate.MetricEvaluateReply\"\x00\x62\x06proto3'
)




_METRICEVALUATEREQUEST = _descriptor.Descriptor(
  name='MetricEvaluateRequest',
  full_name='metricevaluate.MetricEvaluateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='metricevaluate.MetricEvaluateRequest.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='host', full_name='metricevaluate.MetricEvaluateRequest.host', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='start', full_name='metricevaluate.MetricEvaluateRequest.start', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end', full_name='metricevaluate.MetricEvaluateRequest.end', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='algorithm', full_name='metricevaluate.MetricEvaluateRequest.algorithm', index=4,
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
  serialized_start=40,
  serialized_end=138,
)


_METRICEVALUATEREPLY = _descriptor.Descriptor(
  name='MetricEvaluateReply',
  full_name='metricevaluate.MetricEvaluateReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metrics', full_name='metricevaluate.MetricEvaluateReply.metrics', index=0,
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
  serialized_start=140,
  serialized_end=178,
)

DESCRIPTOR.message_types_by_name['MetricEvaluateRequest'] = _METRICEVALUATEREQUEST
DESCRIPTOR.message_types_by_name['MetricEvaluateReply'] = _METRICEVALUATEREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MetricEvaluateRequest = _reflection.GeneratedProtocolMessageType('MetricEvaluateRequest', (_message.Message,), {
  'DESCRIPTOR' : _METRICEVALUATEREQUEST,
  '__module__' : 'metricevaluate_pb2'
  # @@protoc_insertion_point(class_scope:metricevaluate.MetricEvaluateRequest)
  })
_sym_db.RegisterMessage(MetricEvaluateRequest)

MetricEvaluateReply = _reflection.GeneratedProtocolMessageType('MetricEvaluateReply', (_message.Message,), {
  'DESCRIPTOR' : _METRICEVALUATEREPLY,
  '__module__' : 'metricevaluate_pb2'
  # @@protoc_insertion_point(class_scope:metricevaluate.MetricEvaluateReply)
  })
_sym_db.RegisterMessage(MetricEvaluateReply)



_METRICEVALUATESERVICE = _descriptor.ServiceDescriptor(
  name='MetricEvaluateService',
  full_name='metricevaluate.MetricEvaluateService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=180,
  serialized_end=299,
  methods=[
  _descriptor.MethodDescriptor(
    name='MetricEvaluate',
    full_name='metricevaluate.MetricEvaluateService.MetricEvaluate',
    index=0,
    containing_service=None,
    input_type=_METRICEVALUATEREQUEST,
    output_type=_METRICEVALUATEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_METRICEVALUATESERVICE)

DESCRIPTOR.services_by_name['MetricEvaluateService'] = _METRICEVALUATESERVICE

# @@protoc_insertion_point(module_scope)