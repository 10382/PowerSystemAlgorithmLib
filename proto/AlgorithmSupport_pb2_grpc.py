# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto.AlgorithmSupport_pb2 as AlgorithmSupport__pb2


class AlgorithmSupportServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.AlgorithmSupport = channel.unary_unary(
                '/AlgorithmSupport.AlgorithmSupportService/AlgorithmSupport',
                request_serializer=AlgorithmSupport__pb2.AlgorithmSupportRequest.SerializeToString,
                response_deserializer=AlgorithmSupport__pb2.AlgorithmSupportResponse.FromString,
                )


class AlgorithmSupportServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def AlgorithmSupport(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_AlgorithmSupportServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'AlgorithmSupport': grpc.unary_unary_rpc_method_handler(
                    servicer.AlgorithmSupport,
                    request_deserializer=AlgorithmSupport__pb2.AlgorithmSupportRequest.FromString,
                    response_serializer=AlgorithmSupport__pb2.AlgorithmSupportResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'AlgorithmSupport.AlgorithmSupportService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class AlgorithmSupportService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def AlgorithmSupport(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AlgorithmSupport.AlgorithmSupportService/AlgorithmSupport',
            AlgorithmSupport__pb2.AlgorithmSupportRequest.SerializeToString,
            AlgorithmSupport__pb2.AlgorithmSupportResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
