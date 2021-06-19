using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Runtime.InteropServices;

class ENCODER16k{
    [DllImport("libc")]
    static extern int system(String command);	
	//From https://github.com/JDanielSmith/Base16k/blob/master/Base16k/Base16k.cs	
	static string ToBase16KString(byte[] inArray)
	{
		if (inArray == null) throw new ArgumentNullException(nameof(inArray));

		int len = inArray.Length;

		var sb = new StringBuilder(len * 6 / 5);
		sb.Append(len);

		int code = 0;

		for (int i = 0; i < len; ++i)
		{
			byte byteValue = inArray[i];
			switch (i % 7)
			{
				case 0:
					code = byteValue << 6;
					break;

				case 1:
					code |= byteValue >> 2;
					code += 0x5000;
					sb.Append(System.Convert.ToChar(code));
					code = (byteValue & 3) << 12;
					break;

				case 2:
					code |= byteValue << 4;
					break;

				case 3:
					code |= byteValue >> 4;
					code += 0x5000;
					sb.Append(System.Convert.ToChar(code));
					code = (byteValue & 0xf) << 10;
					break;

				case 4:
					code |= byteValue << 2;
					break;

				case 5:
					code |= byteValue >> 6;
					code += 0x5000;
					sb.Append(System.Convert.ToChar(code));
					code = (byteValue & 0x3f) << 8;
					break;

				case 6:
					code |= byteValue;
					code += 0x5000;
					sb.Append(System.Convert.ToChar(code));
					code = 0;
					break;
			}
		}

		// emit a character for remaining bits
		if (len % 7 != 0)
		{
			code += 0x5000;
			sb.Append(System.Convert.ToChar(code));
		}

		return sb.ToString();
	}
	
static void Main(string[] args)
{
string template=
@"using System;
using System.Text;
using System.IO;
using System.Runtime.InteropServices;
class S{
[DllImport(""libc"")]
static extern int system(String c);
static byte C(int c){return System.Convert.ToByte(c);}
static byte[] DEC(string s){
var lE=-1;for(var l=0;l<s.Length;l++)if (s[l]>='0'&&s[l]<='9')lE=l;else break;
int L=Int32.Parse(s.Substring(0,lE+1));var F=new System.Collections.Generic.List<byte>(L);
int P=0;while((P<s.Length)&&(s[P]>='0'&&s[P]<='9'))++P;
int i=0;int X=0;byte bV=0;
while(L-->0)
{if (((1<<i)&0x2b)!=0)X=s[P++]-0x5000;
switch(i%7){
case 0:bV=C(X>>6);F.Add(bV);bV=C((X&0x3f)<<2);break;
case 1:bV|=C(X>>12);F.Add(bV);break;
case 2:bV=C((X>>4)&0xff);F.Add(bV);bV=C((X&0xf)<<4);break;
case 3:bV|=C(X>>10);F.Add(bV);break;
case 4:bV=C((X>>2)&0xff);F.Add(bV);bV=C((X&3)<<6);break;
case 5:bV|=C(X>>8);	F.Add(bV);break;
case 6:bV=C(X&0xff);F.Add(bV);break;
}if (++i==7)i=0;}
return F.ToArray();}

static void Main(){
var s=""CODE"";
System.IO.File.WriteAllBytes(""b.zip"",DEC(s));
system(""unzip b.zip >/dev/null 2>&1;chmod +x Binary >/dev/null 2>&1;./Binary"");
}}";
	
 if (args.Length < 2) {Console.Error.WriteLine("Usage: ENCODER16k.exe <PackedOutput> <binary_executable> <extra_files>");return;}
 string tempZIP = args[0]+".zip";
 system("rm "+args[0]+".zip >/dev/null 2>&1");
 system("zip "+String.Join(' ',args).Replace(args[0],tempZIP)+" >/dev/null 2>&1");
 var CODE = ToBase16KString(File.ReadAllBytes(tempZIP));
 CODE = template.Replace("CODE",CODE).Replace("Binary",args[1]);
 Console.Error.WriteLine("Compressed CG Size:"+CODE.Length+" "+(CODE.Length>999999?" INVALID! ABOVE CG LIMIT OF 100KB!!!!":"OK FOR CODINGAME"));
 File.WriteAllText(args[0],CODE);
 system("rm "+tempZIP+" &>/dev/null");
}

}