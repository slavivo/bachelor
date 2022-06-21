import nmap
# import netifaces

class Ip_Lookup:
    def __init__(self):
        self.__interface = 'None'
        self.__address_family = netifaces.AF_INET
        self.__gateway_address = 'None'
        self.__my_host_ip_address = 'None'
        self.__hosts_ip_addreses = []
    
    def __find_gateway_ip_address(self):
        gateway = netifaces.gateways()['default'][self.__address_family]
        self.__gateway_address=gateway[0]
        self.__interface = gateway[1]
        
    def __find_my_host_ip_address(self):
        ifaddresses = netifaces.ifaddresses(self.__interface)
        self.__my_host_ip_address = ifaddresses[self.__address_family][0]['addr']

    def find_hosts_ip_addresses(self):
        self.__find_gateway_ip_address()
        self.__find_my_host_ip_address()
        nmap_all_hosts = self.__gateway_address + '/24'
        nmap_args = '-n -sn --exclude ' + self.__gateway_address + ',' + self.__my_host_ip_address
        port_scanner = nmap.PortScanner()
        port_scanner.scan(hosts = nmap_all_hosts, arguments = nmap_args)
        for host_ip_address in port_scanner.all_hosts():
            self.__hosts_ip_addreses.append(host_ip_address)
        return self.__hosts_ip_addreses

    class InactiveInterfaceException(Exception):
        pass

    class InterfaceNotConnectedException(Exception):
        pass

    class NoHostsConnectedException(Exception):
        pass
        